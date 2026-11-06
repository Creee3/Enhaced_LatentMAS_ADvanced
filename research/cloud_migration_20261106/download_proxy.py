"""Task-lifetime loopback CONNECT relay; TLS remains end-to-end verified."""
import base64,http.client,http.server,select,socket,urllib.parse,urllib.request

ALLOWED=('pypi.org','pythonhosted.org','pytorch.org','huggingface.co','hf.co',
         'xethub.hf.co','anaconda.com','anaconda.org','conda.io')
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_CONNECT(self):
        host,separator,port=self.path.rpartition(':')
        if not separator or port!='443' or not any(host==x or host.endswith('.'+x) for x in ALLOWED):
            self.send_error(403);return
        upstream=None
        try:
            proxy=urllib.request.getproxies().get('https')
            if proxy:
                parsed=urllib.parse.urlparse(proxy)
                assert parsed.scheme=='http','HTTPS upstream proxy requires a different transport'
                conn=http.client.HTTPConnection(parsed.hostname,parsed.port or 80,timeout=30)
                headers={}
                if parsed.username:
                    auth=urllib.parse.unquote(parsed.username)+':'+urllib.parse.unquote(parsed.password or '')
                    headers['Proxy-Authorization']='Basic '+base64.b64encode(auth.encode()).decode()
                conn.set_tunnel(host,443,headers=headers);conn.connect();upstream=conn.sock
            else:upstream=socket.create_connection((host,443),timeout=30)
            self.send_response(200,'Connection established');self.end_headers()
            upstream.setblocking(False);self.connection.setblocking(False)
            while True:
                readable,_,_=select.select([self.connection,upstream],[],[],120)
                if not readable:break
                for source in readable:
                    data=source.recv(1024*1024)
                    if not data:return
                    target=upstream if source is self.connection else self.connection
                    target.setblocking(True);target.sendall(data);target.setblocking(False)
        except Exception as error:
            print(type(error).__name__+': '+host,flush=True)
        finally:
            if upstream:upstream.close()
    def do_GET(self):self.send_error(405)

if __name__=='__main__':
    print('Temporary loopback package/model TLS relay on 127.0.0.1:17863',flush=True)
    http.server.ThreadingHTTPServer(('127.0.0.1',17863),Handler).serve_forever()
