import http.client

conn = http.client.HTTPConnection("localhost", 8000)
conn.request("POST", "/")
html_response = conn.getresponse()

print(html_response, html_response.reason)

html_data = html_response.read()
print(html_data)