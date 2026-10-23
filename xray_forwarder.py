import socket, json, boto3, os
region = os.environ.get("AWS_REGION", "us-east-1")  # set to config.AWS_REGION
xray = boto3.client("xray", region_name=region)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("127.0.0.1", 2000))
print("listening udp/2000")
while True:
    data, _ = s.recvfrom(65535)
    try:
        _, doc = data.decode().split("\n", 1)
        json.loads(doc)
        xray.put_trace_segments(TraceSegmentDocuments=[doc])
    except Exception as e:
        print("ERR", e)