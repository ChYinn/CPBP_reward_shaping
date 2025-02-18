package minicpbp.examples;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.Arrays;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import minicpbp.util.exception.InconsistencyException;
import minicpbp.util.io.InputReader;


public class AI_Front_Train {
   private static String directory = "./src/main/java/minicpbp/examples/data/ClassicalAIplanning/";
   private static AI_Instance_Manager ai_manager;
   public static void generate_results(){
      
   }
   public static void main(String[] args) throws IOException {       
      HttpServer server = HttpServer.create(new InetSocketAddress(   9005), 0);
      ai_manager = new AI_Instance_Manager(directory);
      //ai_manager.load();
      server.createContext("/solve", new MyHandler());
      server.setExecutor(null); // creates a default executor
      server.start();
   }
   static class MyHandler implements HttpHandler {
      @Override
      public void handle(HttpExchange t) throws IOException {
         String response;
         try {
            response = "Loaded";
            /*System.out.println("Trying...");
                if(ai_manager.plan != null){
                for (int i = 0; i < ai_manager.plan.action.length; i++) {
                       System.ou t.print(Integer.toString(i));
                       System.out.print(": ");
                       System.out.print(ai_manager.plan.action[i].toString());
                       System.out.print(", ");
                   }
                System.out.println();
            }*/
            ai_manager.load();
            
         } catch (Exception e) {
            System.out.println(e);
            if(e instanceof InconsistencyException){
               //System.out.println("Inconsistency exception");
               ai_manager.require_restart = true;
               /*for (int i = 0; i < ai_manager.plan.action.length; i++) {
                   System.out.print(Integer.toString(i));
                   System.out.print(": ");
                   System.out.print(ai_manager.plan.action[i].toString());
                   System.out.print(", ");
               }
               System.out.println();*/
               /*
               for (int i = 0; i < ai_manager.plan.action.length; i++) {
                if(ai_manager.plan.action[i].isBound()){
                    System.out.print(ai_manager.plan.action[i].min());
                }
                else{
                    System.out.print("{");
                    for (int j = ai_manager.plan.action[i].min(); j <= ai_manager.plan.action[i].max(); j++) {
                        System.out.print(j);
                        System.out.print(", ");
                    }
                    System.out.print("}, ");
                }
                System.out.print(", ");
            }
            System.out.println();*/
               
               
               response = "inconsistency";
               InputReader reader = new InputReader(directory+"parameters.txt");
               int planSize = reader.getInt();          //System.out.println("planSize: "+Integer.toString(planSize));
               int nBlocks  = reader.getInt();          //System.out.println("nBlocks: "+Integer.toString(nBlocks));
               int nSteps   = reader.getInt();          //System.out.println("nSteps: "+Integer.toString(nSteps));
               int[] action;
               if(nSteps>0){
                     Integer[] action_ = reader.getIntLine(); 
                     action = new int[nSteps];
                     for (int i = 0; i < nSteps; i++) {
                        action[i]=action_[i].intValue();
                     }
                     //System.out.println("action: "+Arrays.toString(action));
               }
               else{
                     action = new int[]{};
               }
               Integer[] configuration_init_ = reader.getIntLine();
               int[] configuration_init = new int[nBlocks];
               for (int i = 0; i < nBlocks; i++) {
                     configuration_init[i]=configuration_init_[i].intValue();
               }
               //System.out.println("configuration_init: "+Arrays.toString(configuration_init));
               Integer[] configuration_target_  = reader.getIntLine();
               int[] configuration_target = new int[nBlocks];
               for (int i = 0; i < nBlocks; i++) {
                     configuration_target[i]=configuration_target_[i].intValue();
               }
               //System.out.println("[~~~~~~~~~~~~~~~~~~~~~]");
               //System.out.println("planSize: "+Integer.toString(planSize));
               //System.out.println("nBlocks: "+Integer.toString(nBlocks));
               //System.out.println("nSteps: "+Integer.toString(nSteps));
               //System.out.println("configuration_init: "+Arrays.toString(configuration_init));
               //System.out.println("configuration_target: "+Arrays.toString(configuration_target));
               //System.out.println("actions: "+Arrays.toString(action));
            }
            else{
               InputReader reader = new InputReader(directory+"parameters.txt");
               int planSize = reader.getInt();          //System.out.println("planSize: "+Integer.toString(planSize));
               int nBlocks  = reader.getInt();          //System.out.println("nBlocks: "+Integer.toString(nBlocks));
               int nSteps   = reader.getInt();          //System.out.println("nSteps: "+Integer.toString(nSteps));
               int[] action;
               if(nSteps>0){
                     Integer[] action_ = reader.getIntLine(); 
                     action = new int[nSteps];
                     for (int i = 0; i < nSteps; i++) {
                        action[i]=action_[i].intValue();
                     }
                     //System.out.println("action: "+Arrays.toString(action));
               }
               else{
                     action = new int[]{};
               }
               Integer[] configuration_init_ = reader.getIntLine();
               int[] configuration_init = new int[nBlocks];
               for (int i = 0; i < nBlocks; i++) {
                     configuration_init[i]=configuration_init_[i].intValue();
               }
               //System.out.println("configuration_init: "+Arrays.toString(configuration_init));
               Integer[] configuration_target_  = reader.getIntLine();
               int[] configuration_target = new int[nBlocks];
               for (int i = 0; i < nBlocks; i++) {
                     configuration_target[i]=configuration_target_[i].intValue();
               }
               response = e.getMessage()+"\n"+e.getLocalizedMessage();
               response += "planSize: "+Integer.toString(planSize)+"\n";
               response +="nBlocks: "+Integer.toString(nBlocks)+"\n";
               response +="nSteps: "+Integer.toString(nSteps)+"\n";
               response +="configuration_init: "+Arrays.toString(configuration_init)+"\n";
               response +="configuration_target: "+Arrays.toString(configuration_target)+"\n";
               response +="actions: "+Arrays.toString(action)+"\n";
            }
            
            
         }
         ai_manager.write_results();
         t.sendResponseHeaders(200, response.length());
         OutputStream os = t.getResponseBody();
         os.write(response.getBytes());
         os.close();
      }
   }
}
