package minicpbp.examples;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;


public class AI_Front_Solve {
   private static String directory;

     private static void solve(){
        ClassicalAIplanning problem_instance = new ClassicalAIplanning();
        problem_instance.solve();

     }
     public static void main(String[] args) throws IOException {       
        HttpServer server = HttpServer.create(new InetSocketAddress(9005), 0);
        server.createContext("/solve", new MyHandler());
        server.setExecutor(null); // creates a default executor
        server.start();
     }
     static class MyHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            //
            solve();
            String response = "Solved";
            t.sendResponseHeaders(200, response.length());
            OutputStream os = t.getResponseBody();
            os.write(response.getBytes());
            os.close();
        }
    }
}
