import boto3, config
c = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
rid = config.AGENTCORE_RUNTIME_ARN.split('/')[-1]
r = c.get_agent_runtime(agentRuntimeId=rid)
env = dict(r.get('environmentVariables', {}))
env.update({
    'AGENT_LOG_GROUP': config.AGENT_LOG_GROUP,
    'AGENT_LOG_LEVEL': 'INFO',
    'AGENT_LOG_TO_CLOUDWATCH': 'true',
    'AGENT_TRACING_ENABLED': 'true',
    'AGENT_TRACE_SAMPLING_RATE': '1.0',
})
kw = dict(agentRuntimeId=rid, agentRuntimeArtifact=r['agentRuntimeArtifact'],
          roleArn=r['roleArn'], networkConfiguration=r['networkConfiguration'],
          environmentVariables=env)
if r.get('protocolConfiguration'):
    kw['protocolConfiguration'] = r['protocolConfiguration']
print(c.update_agent_runtime(**kw)['status'])