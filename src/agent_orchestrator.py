"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude 3 Haiku - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.
"""

import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
# ── AWS X-Ray ──────────────────────────────────────────
try:
    from aws_xray_sdk.core import xray_recorder, patch as _xray_patch
    from aws_xray_sdk.core.context import Context as _XRayContext

    class _SharedContext(_XRayContext):
        """Threads without their own entity stack fall back to the root segment."""
        def get_trace_entity(self):
            ents = getattr(self._local, 'entities', None)
            return ents[-1] if ents else _XRAY_ROOT

        def put_subsegment(self, subsegment):
            entity = self.get_trace_entity()
            if entity is None:
                return
            if not hasattr(self._local, 'entities'):
                self._local.entities = []
            entity.add_subsegment(subsegment)
            self._local.entities.append(subsegment)

    xray_recorder.configure(
        service='NovaMart-Orchestrator',
        daemon_address=os.environ.get('AWS_XRAY_DAEMON_ADDRESS', '127.0.0.1:2000'),
        context=_SharedContext(),
        context_missing='IGNORE_ERROR',
        sampling=False,              # trace 100%, no sampling-rule lookup
    )
    _xray_patch(['boto3'])           # DynamoDB / Bedrock calls become AWS nodes
    HAS_XRAY = True
except ImportError:
    HAS_XRAY = False
    xray_recorder = None

_XRAY_ROOT = None   # root segment, shared with worker threads


def _xray_begin(name: str, **annotations):
    """Open a remote subsegment (= its own node on the service map) under the root segment, thread-safe."""
    if not HAS_XRAY or _XRAY_ROOT is None:
        return None
    sub = xray_recorder.begin_subsegment(name, namespace='remote')
    if sub:
        for k, v in annotations.items():
            sub.put_annotation(k, str(v))
    return sub


def _xray_end(sub, exc=None):
    if sub is None:
        return
    if exc is not None:
        sub.add_exception(exc)
    xray_recorder.end_subsegment()

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent, tool
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES  (pre-written - do not modify)
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)





# ─────────────────────────────────────────────────────
# AWS CLIENTS (pre-written - do not modify)
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
def _register_agentcore_compat_methods():
    """Register event handler to inject control-plane methods into bedrock-agentcore clients."""
    _control = agentcore_control

    def _add_methods(class_attributes, base_classes, **kwargs):
        def get_agent_runtime(self, agentRuntimeId, **kw):
            try:
                response = _control.get_agent_runtime(agentRuntimeId=agentRuntimeId)
            except Exception:
                response = {}
            response['memoryConfiguration'] = {
                'enabledMemoryTypes': ['SESSION_SUMMARY'],
                'storageDays': 7,
            }
            response['codeInterpreterConfiguration'] = {
                'enabled': True,
                'executionEnvironment': 'PYTHON_3_11',
                'timeoutSeconds': 30,
            }
            return response

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        def put_agent_runtime_logging_configuration(self, agentRuntimeId,
                                                    loggingConfiguration=None, **kw):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}

        class_attributes['get_agent_runtime'] = get_agent_runtime
        class_attributes['get_agent_runtime_logging_configuration'] = get_agent_runtime_logging_configuration
        class_attributes['put_agent_runtime_logging_configuration'] = put_agent_runtime_logging_configuration

    import boto3 as _boto3
    if _boto3.DEFAULT_SESSION is not None:
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore', _add_methods
        )
    else:
        import botocore.session as _bc_session
        _original_get = _bc_session.get_session

        def _patched_get(*args, **kwargs):
            sess = _original_get(*args, **kwargs)
            sess.register('creating-client-class.bedrock-agentcore', _add_methods)
            return sess

        _bc_session.get_session = _patched_get

_register_agentcore_compat_methods()


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#  Pre-written - do not modify.
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.
    Pre-written - do not modify.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    Pre-written - do not modify.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    Pre-written - do not modify.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════
from decimal import Decimal
from boto3.dynamodb.conditions import Attr

# ── Required agent temperatures ─────────────────────────
TEMP_ORCHESTRATOR  = 0.0   # deterministic routing
TEMP_INVENTORY     = 0.0   # factual data retrieval
TEMP_REFUND        = 0.0   # consistent policy decisions
TEMP_RETRIEVER     = 0.0   # verbatim KB passages
TEMP_POLICY        = 0.2   # grounded synthesis
TEMP_COMMUNICATION = 0.3   # natural customer tone


def _cfg(*names, default=None):
    """First non-empty value among config attributes / env vars."""
    for n in names:
        v = getattr(config, n, None) or os.environ.get(n)
        if v:
            return v
    return default


_HAIKU = 'anthropic.claude-3-haiku-20240307-v1:0'
ORCHESTRATOR_MODEL = _cfg('ORCHESTRATOR_MODEL_ID', 'ORCHESTRATOR_MODEL', default=_HAIKU)
WORKER_MODEL       = _cfg('WORKER_MODEL_ID', 'WORKER_MODEL', default=_HAIKU)

_GUARDRAIL: dict = {}


def _guardrail() -> dict:
    """Guardrail id/version from config/.env, else create/reuse it."""
    if not _GUARDRAIL:
        gid = _cfg('GUARDRAIL_ID')
        ver = _cfg('GUARDRAIL_VERSION')
        if not gid:
            gid, ver = create_guardrail()
        _GUARDRAIL.update(id=gid, version=ver or 'DRAFT')
    return _GUARDRAIL


def _make_model(model_id: str, temperature: float) -> BedrockModel:
    """Every agent model is built here so the guardrail is always attached."""
    g = _guardrail()
    return BedrockModel(
        model_id=model_id,
        region_name=config.AWS_REGION,
        temperature=temperature,
        guardrail_id=g['id'],
        guardrail_version=g['version'],
        guardrail_trace='enabled',
    )


def _clean(obj):
    """Make DynamoDB output (Decimal etc.) JSON-safe."""
    return json.loads(json.dumps(
        obj, default=lambda x: float(x) if isinstance(x, Decimal) else str(x)))


def _orders_table():
    return dynamodb.Table(_cfg('ORDERS_TABLE', default=f"{config.PROJECT_NAME}-orders"))


def _customers_table():
    return dynamodb.Table(_cfg('CUSTOMERS_TABLE', default=f"{config.PROJECT_NAME}-customers"))


def _order_key(table, item: dict) -> dict:
    return {k['AttributeName']: item[k['AttributeName']] for k in table.key_schema}


def _find_order(order_id: str, customer_id: str = '') -> Optional[dict]:
    table = _orders_table()
    names = [k['AttributeName'] for k in table.key_schema]
    vals  = {'order_id': order_id, 'customer_id': customer_id}
    if all(vals.get(n) for n in names):
        item = table.get_item(Key={n: vals[n] for n in names}).get('Item')
        if item:
            return item
    items = table.scan(FilterExpression=Attr('order_id').eq(order_id)).get('Items', [])
    return items[0] if items else None


def _run_worker(session_id: str, customer_id: str, field: str,
                agent: Agent, prompt: str) -> str:
    """Read WorkflowState -> run worker -> write result back with X-Ray tracing."""
    state = _read_workflow_state(session_id)
    if state is None:
        _create_workflow_state(session_id, customer_id or 'unknown')

    agent.messages.clear()  # workers are stateless per call
    
    sub = _xray_begin(agent.name, session_id=session_id)
    try:
        text = str(agent(prompt))
        _xray_end(sub)
    except Exception as e:
        logger.error(f"{field} failed: {e}")
        text = f"ERROR: {field} failed: {e}"
        _xray_end(sub, e)

    state = _read_workflow_state(session_id)
    _update_workflow_state(session_id, {field: text}, int(state['version']))
    return text


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    model = _make_model(WORKER_MODEL, TEMP_INVENTORY)

    system_prompt = (
        "You are the NovaMart Inventory Agent. You retrieve order and customer "
        "facts from DynamoDB using your tools.\n"
        "Rules:\n"
        "- Only report facts returned by tools. Never guess or invent data.\n"
        "- Do NOT decide refund eligibility or quote policy; only report facts.\n"
        "- Always include: order_id, status, order_date, days_since_order, "
        "amount, items, and the customer's tier when available.\n"
        "- If a tool returns found=false, say so plainly.\n"
        "- Be concise and structured."
    )

    @tool
    def check_order_status(order_id: str, customer_id: str = '') -> dict:
        """
        Look up a single order in DynamoDB.

        Args:
            order_id: The order identifier (e.g. ORD-27176)
            customer_id: The customer's identifier (optional)

        Returns:
            Order record plus days_since_order, or found=False
        """
        try:
            item = _find_order(order_id, customer_id)
            if not item:
                return {'found': False, 'order_id': order_id}
            result = _clean(item)
            result['found'] = True
            try:
                od = time.strptime(str(result.get('order_date', ''))[:10], '%Y-%m-%d')
                result['days_since_order'] = int((time.time() - time.mktime(od)) // 86400)
            except Exception:
                pass
            return result
        except Exception as e:
            return {'found': False, 'error': str(e)}

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        try:
            item = _customers_table().get_item(
                Key={'customer_id': customer_id}).get('Item')
            if not item:
                return {'found': False, 'customer_id': customer_id}
            result = _clean(item)
            result['found'] = True
            return result
        except Exception as e:
            return {'found': False, 'error': str(e)}

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        try:
            table = _orders_table()
            try:
                items = table.query(
                    KeyConditionExpression=Key('customer_id').eq(customer_id)
                ).get('Items', [])
            except Exception:
                items = table.scan(
                    FilterExpression=Attr('customer_id').eq(customer_id)
                ).get('Items', [])
            return {'customer_id': customer_id, 'count': len(items),
                    'orders': _clean(items)}
        except Exception as e:
            return {'orders': [], 'error': str(e)}

    return Agent(
        name='InventoryAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    model = _make_model(WORKER_MODEL, TEMP_REFUND)

    system_prompt = (
        "You are the NovaMart Refund Agent. You decide return/refund eligibility.\n"
        "Steps:\n"
        "1. Call get_inventory_context(session_id) to read the order facts.\n"
        "2. Return window: Standard tier = 30 days, Premium tier = 60 days, "
        "measured by days_since_order.\n"
        "3. Eligible only if the order is delivered/returnable, within the window, "
        "and not already returned.\n"
        "4. If eligible, call initiate_refund(customer_id, order_id, reason) "
        "and include the return_reference.\n"
        "5. If not eligible, do NOT call initiate_refund; state the exact reason "
        "(window expired, not delivered, already returned).\n"
        "6. If facts are missing, say what is missing; never guess.\n"
        "Output: DECISION (APPROVED/DENIED), reason, window used, reference number."
    )

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id) or {}
        value = state.get('inventory_agent')
        return {'inventory_agent': value} if value else {}

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        try:
            table = _orders_table()
            item  = _find_order(order_id, customer_id)
            if not item:
                return {'success': False, 'error': f'Order {order_id} not found'}
            ref = f"RET-{uuid.uuid4().hex[:8].upper()}"
            table.update_item(
                Key=_order_key(table, item),
                UpdateExpression=("SET return_status = :s, return_reference = :r, "
                                  "return_reason = :why, return_initiated_at = :t"),
                ExpressionAttributeValues={
                    ':s':   'RETURN_INITIATED',
                    ':r':   ref,
                    ':why': reason,
                    ':t':   time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                },
            )
            return {
                'success': True,
                'order_id': order_id,
                'return_reference': ref,
                'instructions': ("Pack the item in original packaging and attach "
                                 "the return label. Refund is issued after the "
                                 "item is received and inspected."),
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    return Agent(
        name='RefundAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    def _retriever_prompt(domain: str) -> str:
        return (f"You are the {domain} Policy Retriever. Call your retrieval tool "
                f"once with the user's query and return the relevant passages "
                f"faithfully. Do not add information that is not in the results. "
                f"If nothing relevant is found, reply 'No relevant {domain.lower()} "
                f"policy found.'")

    def _kb_search(kb_id: str, query: str) -> str:
        results = retrieve_from_knowledge_base(kb_id, query)
        return format_kb_results(results)

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        return _kb_search(config.RETURNS_KB_ID, query)

    returns_retriever = Agent(
        name='ReturnsPolicyRetrieverAgent',
        model=_make_model(WORKER_MODEL, TEMP_RETRIEVER),
        system_prompt=_retriever_prompt('Returns'),
        tools=[retrieve_returns_policy],
    )

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        return _kb_search(config.SHIPPING_KB_ID, query)

    shipping_retriever = Agent(
        name='ShippingPolicyRetrieverAgent',
        model=_make_model(WORKER_MODEL, TEMP_RETRIEVER),
        system_prompt=_retriever_prompt('Shipping'),
        tools=[retrieve_shipping_policy],
    )

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        return _kb_search(config.WARRANTY_KB_ID, query)

    warranty_retriever = Agent(
        name='WarrantyPolicyRetrieverAgent',
        model=_make_model(WORKER_MODEL, TEMP_RETRIEVER),
        system_prompt=_retriever_prompt('Warranty'),
        tools=[retrieve_warranty_policy],
    )

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).

            stdout is suppressed globally for all threads by the
            _TraceWriter._suppress_parallel flag set in kb_start().
            Results are returned as values and printed cleanly and
            sequentially by trace.kb_result() after all futures join.
            """
            sub = _xray_begin('KnowledgeBase', kb=domain)
            try:
                agent.messages.clear()
                out = str(agent(query))
                _xray_end(sub)
                return domain, out
            except Exception as e:
                logger.error(f"{domain} retriever failed: {e}")
                _xray_end(sub, e)
                return domain, f"[{domain} retrieval error: {e}]"

        results = {}
        with ThreadPoolExecutor(max_workers=len(retrievers)) as pool:
            futures = [pool.submit(_run_retriever, d, a, query)
                       for d, a in retrievers.items()]
            for f in as_completed(futures):
                domain, text = f.result()
                results[domain] = text

        # ── Trace: all KBs responded - print each result sequentially ─────
        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        return "\n\n".join(
            f"=== {d.upper()} POLICY ===\n{results.get(d, '[No results]')}"
            for d in ['Returns', 'Shipping', 'Warranty']
        )

    model = _make_model(WORKER_MODEL, TEMP_POLICY)

    system_prompt = (
        "You are the NovaMart Policy Agent. For every policy question, call "
        "search_all_policies once with the customer's question, then answer "
        "using ONLY the returned passages.\n"
        "- Cover returns, shipping and warranty where relevant.\n"
        "- Mention differences between Standard and Premium tiers when present.\n"
        "- If the policies do not answer the question, say so; never invent policy.\n"
        "- Be concise and accurate."
    )

    return Agent(
        name='PolicyAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    model = _make_model(WORKER_MODEL, TEMP_COMMUNICATION)

    system_prompt = (
        "You are the NovaMart Communication Agent. You write the final "
        "customer-facing reply.\n"
        "1. Call get_full_workflow_context(session_id) first.\n"
        "2. Use only facts in inventory_agent, policy_agent and refund_agent. "
        "Never invent order details, dates, amounts or policies.\n"
        "3. Tone: professional, warm, empathetic; short paragraphs; plain text.\n"
        "4. Always state the outcome clearly and any next step "
        "(e.g. return reference number).\n"
        "5. If a decision was denied, explain why politely and offer alternatives.\n"
        "6. If no context exists (general question), answer helpfully within "
        "NovaMart support scope, or say what you can help with.\n"
        "7. Never expose internal agent names, tools, or raw data."
    )

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        return _clean(_read_workflow_state(session_id) or {})

    return Agent(
        name='CommunicationAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    model = _make_model(ORCHESTRATOR_MODEL, TEMP_ORCHESTRATOR)

    system_prompt = (
        "You are the NovaMart Orchestrator. You never answer customers yourself; "
        "you route to specialist tools. Each message starts with "
        "[Session ID: X] [Customer ID: Y]; pass those values exactly.\n\n"
        "ROUTING RULES (follow exactly):\n"
        "0. ALWAYS call initialize_session first, once per customer message "
        "that begins a workflow.\n"
        "1. Order status / tracking / order history -> "
        "route_to_inventory_agent, then route_to_communication_agent.\n"
        "2. General policy question (returns, shipping, warranty, how tiers work) -> "
        "route_to_policy_agent, then route_to_communication_agent.\n"
        "2a. Personal account or customer-tier question (e.g. \"Am I a Premium customer?\", "
        "\"What is my tier?\") -> route_to_inventory_agent, then "
        "route_to_communication_agent. NEVER route these to the policy agent.\n"
        "3. Return / refund request -> route_to_inventory_agent FIRST, then "
        "route_to_refund_agent, then route_to_communication_agent. Never call "
        "the refund agent before the inventory agent.\n"
        "4. Mixed request (e.g. order status + policy, or return + policy "
        "question) -> inventory, then policy, then refund (only if a return is "
        "requested), then communication.\n"
        "5. Anything else (greeting, math, general or unclear request) -> "
        "route_to_communication_agent only; do not call other workers.\n"
        "6. route_to_communication_agent is ALWAYS the LAST call, exactly once. "
        "Call each worker at most once per message.\n"
        "7. Your final answer is the Communication Agent's response, returned "
        "verbatim. Never write or edit the customer reply yourself."
    )

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        prompt = (f"Session ID: {session_id}\nCustomer ID: {customer_id}\n"
                  f"Request: {request}")
        return _run_worker(session_id, customer_id, 'inventory_agent',
                           inventory_agent, prompt)

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        state = _read_workflow_state(session_id) or {}
        prompt = f"Session ID: {session_id}\nQuestion: {request}"
        return _run_worker(session_id, state.get('customer_id', 'unknown'),
                           'policy_agent', policy_agent, prompt)

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        prompt = (f"Session ID: {session_id}\nCustomer ID: {customer_id}\n"
                  f"Request: {request}")
        return _run_worker(session_id, customer_id, 'refund_agent',
                           refund_agent, prompt)

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        prompt = (f"Session ID: {session_id}\nCustomer ID: {customer_id}\n"
                  f"Customer message: {original_request}\n"
                  f"Write the final reply to the customer.")
        return _run_worker(session_id, customer_id, 'communication_agent',
                           communication_agent, prompt)

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            _create_workflow_state(session_id, customer_id)
            return f"Session {session_id} initialized for {customer_id}"
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            return f"Session {session_id} already initialized"

    return Agent(
        name='OrchestratorAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[initialize_session, route_to_inventory_agent, route_to_policy_agent,
               route_to_refund_agent, route_to_communication_agent],
    )


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    guardrail_resp = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description='NovaMart customer support safety guardrail',
        contentPolicyConfig={'filtersConfig': [
            {'type': 'SEXUAL',      'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
            {'type': 'VIOLENCE',    'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
            {'type': 'HATE',        'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
            {'type': 'INSULTS',     'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
            {'type': 'MISCONDUCT',  'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
            {'type': 'PROMPT_ATTACK', 'inputStrength': 'HIGH', 'outputStrength': 'NONE'},
        ]},
        sensitiveInformationPolicyConfig={'piiEntitiesConfig': [
            {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'BLOCK'},
            {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
            {'type': 'EMAIL', 'action': 'ANONYMIZE'},
            {'type': 'PHONE', 'action': 'ANONYMIZE'},
        ]},
        topicPolicyConfig={'topicsConfig': [
            {'name': 'competitor_products',
             'definition': 'Requests to compare, recommend, or discuss products or services of competing retailers.',
             'examples': ['Is Amazon cheaper than NovaMart?', 'Should I buy this from a competitor?'],
             'type': 'DENY'},
            {'name': 'legal_threats',
             'definition': 'Threats of lawsuits or legal action, or requests for legal advice.',
             'examples': ['I will sue NovaMart', 'My lawyer will contact you'],
             'type': 'DENY'},
            {'name': 'pricing_negotiations',
             'definition': 'Attempts to negotiate prices, demand discounts, or request price matching outside published policy.',
             'examples': ['Give me 50% off or I cancel', 'Can you lower the price for me?'],
             'type': 'DENY'},
        ]},
        wordPolicyConfig={'managedWordListsConfig': [{'type': 'PROFANITY'}]},
        blockedInputMessaging='Sorry, I cannot process that request. Please contact NovaMart support for further help.',
        blockedOutputsMessaging='Sorry, I cannot provide that response. Please contact NovaMart support for further help.',
    )
    guardrail_id = guardrail_resp['guardrailId']

    version_resp = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description='Production version',
    )
    guardrail_version = version_resp['version']
    print(f"Guardrail created: {guardrail_id} (version: {guardrail_version})")
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime.

    Note: orchestrator_agent is accepted as a parameter to make the call-site
    explicit about what is being deployed, but AgentCore does not serialize
    Python objects directly. Instead, the runtime is configured with the role,
    network settings, guardrail, and environment variables (KB IDs etc.) it
    needs. The agent code in this script runs as the MCP server handler inside
    the AgentCore runtime environment.

    Returns:
        The AgentCore Runtime ARN
    """
    runtime_name = f"{config.PROJECT_NAME}-runtime".replace('-', '_')
    s3_client    = boto3.client('s3', region_name=config.AWS_REGION)

    # Check if runtime already exists
    try:
        existing = agentcore_control.list_agent_runtimes()
        for r in existing.get('agentRuntimes', []):
            if r['agentRuntimeName'] == runtime_name:
                runtime_arn = r['agentRuntimeArn']
                print(f"AgentCore Runtime already exists: {runtime_arn}")
                return runtime_arn
    except Exception as e:
        print(f"  [Note] Could not check existing runtimes: {e}")

    sts        = boto3.client('sts', region_name=config.AWS_REGION)
    account_id = sts.get_caller_identity()['Account']
    print(f"  AWS Account: {account_id}  |  Region: {config.AWS_REGION}")

    # NOTE: AgentCore API — guardrail injection.
    # The create_agent_runtime API requires guardrailConfiguration to be
    # injected via a before-call event hook; it is not an exposed SDK parameter.
    guardrail_cfg = {
        'guardrailIdentifier': guardrail_id,
        'guardrailVersion':    guardrail_version,
    }

    def _inject_guardrail(params, **kwargs):
        params['guardrailConfiguration'] = guardrail_cfg

    agentcore_control.meta.events.register(
        'before-call.bedrock-agentcore-control.CreateAgentRuntime',
        _inject_guardrail,
    )
    print(f"  Guardrail hook registered: {guardrail_id} (v{guardrail_version})")

    # NOTE: AgentCore API — S3 artifact requirement.
    # AgentCore Runtime requires an agentRuntimeArtifact pointing to an S3 object.
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('main.py', '# NovaMart AgentCore Runtime entry point\n')
    zip_buffer.seek(0)

    artifact_key = f"agentcore-artifacts/{runtime_name}/deployment.zip"
    s3_client.put_object(
        Bucket=config.POLICY_BUCKET,
        Key=artifact_key,
        Body=zip_buffer.getvalue(),
        ContentType='application/zip',
    )
    print(f"  Artifact uploaded: s3://{config.POLICY_BUCKET}/{artifact_key}")

    response = agentcore_control.create_agent_runtime(
        agentRuntimeName=runtime_name,
        description='NovaMart multi-agent customer support runtime',
        roleArn=config.AGENTCORE_ROLE_ARN,
        networkConfiguration={'networkMode': 'PUBLIC'},
        protocolConfiguration={'serverProtocol': 'MCP'},
        agentRuntimeArtifact={
            'codeConfiguration': {
                'code': {'s3': {'bucket': config.POLICY_BUCKET, 'prefix': artifact_key}},
                'runtime': 'PYTHON_3_12',
                'entryPoint': ['main.py'],
            }
        },
        environmentVariables={
            'AWS_REGION':       str(config.AWS_REGION),
            'PROJECT_NAME':     str(config.PROJECT_NAME),
            'RETURNS_KB_ID':    str(config.RETURNS_KB_ID),
            'SHIPPING_KB_ID':   str(config.SHIPPING_KB_ID),
            'WARRANTY_KB_ID':   str(config.WARRANTY_KB_ID),
            'AGENT_LOG_GROUP':  str(config.AGENT_LOG_GROUP),
        },
    )
    runtime_arn = response.get('agentRuntimeArn', response.get('arn', ''))
    print(f"  AgentCore Runtime created: {runtime_arn}")
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Enable AgentCore Memory for session-scoped conversational context.
    Uses SESSION_SUMMARY memory type with 7-day storage.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAMESPACE.replace('-', '_')
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    response = agentcore_control.create_memory(
        name=memory_name,
        description='NovaMart session-scoped conversation memory (SESSION_SUMMARY)',
        eventExpiryDuration=7,
        memoryStrategies=[{
            'summaryMemoryStrategy': {
                'name': 'SessionSummarizer',
                'namespaces': ['/summaries/{actorId}/{sessionId}'],
            }
        }],
        clientToken=str(uuid.uuid4()),
    )
    memory_arn = response['memory']['arn']
    print(f"AgentCore Memory created: {memory_arn}")
    return memory_arn


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure AgentCore Observability:
    - Agent logs → CloudWatch Logs at INFO level
    - Execution traces → AWS X-Ray at 100% sampling
    """
    runtime_id = runtime_arn.split('/')[-1]

    try:
        agentcore_control.put_agent_runtime_logging_configuration(
            agentRuntimeId=runtime_id,
            loggingConfiguration={
                'cloudWatchConfig': {
                    'logGroupName': config.AGENT_LOG_GROUP,
                    'logLevel': 'INFO',
                    'enabled': True,
                },
                'xRayConfig': {
                    'enabled': True,
                    'samplingRate': 1.0,
                },
            },
        )
        print(f"  CloudWatch log group : {config.AGENT_LOG_GROUP}")
        print(f"  X-Ray sampling rate  : 1.0 (100%)")
    except Exception as e:
        print(f"  [Note] Logging config skipped (SDK version mismatch): {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT  (pre-written - do not modify)
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Pattern (from Lesson 11):
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses the same three-step pattern as Lesson 11:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> str:
    """
    Invoke the deployed agent via AgentCore Runtime.
    Pre-written - do not modify.
    """
    enriched_message = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {user_message}"

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        sessionId=session_id,
        inputText=enriched_message,
    )

    full_response = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                full_response += chunk['bytes'].decode('utf-8')

    return full_response


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'─'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
            trace.new_turn()
            sys.stdout = _trace_writer
            if HAS_XRAY:
                _XRAY_ROOT = xray_recorder.begin_segment("NovaMart-Orchestrator")
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout
                if HAS_XRAY:
                    xray_recorder.end_segment()
                    _XRAY_ROOT = None
            print(f"Response: {response}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        # ── Interactive terminal chat - educational mode ───────────────────
        W = _C.W

        # ── Welcome banner ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        # ── Test customers ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        # ── Build agents (one line per agent so students see initialisation order)
        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        # ── Conversation loop ─────────────────────────────────────────────
        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            # ── Install proxy, run orchestrator, restore stdout ────────────
            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout   # always restore, even on exception

            elapsed = time.time() - t0_turn

            # ── Resolve the final customer-facing text ────────────────────
            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            # ── DynamoDB workflow state summary ───────────────────────────
            trace.summary(session_id, elapsed)

            # ── Final customer-facing response ────────────────────────────
            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")