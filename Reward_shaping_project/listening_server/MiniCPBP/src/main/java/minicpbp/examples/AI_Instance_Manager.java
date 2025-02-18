package minicpbp.examples;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;

import minicpbp.util.io.InputReader;

public class AI_Instance_Manager {
    private String directory;
    public boolean require_restart;
    public AI_Planning_Instance plan;
    public AI_Instance_Manager(String directory_){
        this.directory=directory_;
        require_restart = false;
    }
    private boolean matchPlan(int planSize_read, int[] action, int[] configuration_init_, int[] configuration_target_){
        
        if(this.plan==null) return false;
        else if(this.plan.planSize != planSize_read) return false;
        else if(this.plan.currentIndex+this.plan.nStays>action.length) return false;
        else{
            if(configuration_init_.length   != plan.configuration_init.length) return false;
            if(configuration_target_.length != plan.configuration_target.length) return false;

            for (int i = 0; i < this.plan.currentIndex+this.plan.nStays; i++) {
                if(this.plan.previousActions[i]!=action[i]) return false;
            }
            for (int i = 0; i < configuration_init_.length; i++){
                if(plan.configuration_init[i] != configuration_init_[i]) return false;
            }
            for (int i = 0; i < configuration_target_.length; i++){
                if(plan.configuration_target[i] != configuration_target_[i]) return false;
            }

            
            return true;
        }
    }

    public void load(){
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
        Integer[] configuration_target_  = reader.getIntLine();
        int[] configuration_target = new int[nBlocks];
        for (int i = 0; i < nBlocks; i++) {
            configuration_target[i]=configuration_target_[i].intValue();
        }
        double threshold   = reader.getDoubleLine()[0];
        Double[] distribution = reader.getDoubleLine();
        require_restart = true;
        if(!matchPlan(planSize, action, configuration_init, configuration_target) || require_restart){
            this.plan = new AI_Planning_Instance(this.directory, planSize, nBlocks, configuration_init, configuration_target);
            this.plan.threshold = threshold;
            require_restart = false;
        }
        this.plan.setActions(action);
        this.plan.propagate();
    }
    public double get_expected_cost(){
        return this.plan.get_expected_cost();
    }
    public int get_minimum_cost(){
        return this.plan.get_minimum_cost();
    }
    public double get_entropy(){
        return this.plan.get_entropy();
    }
    public void write_results(){
        double expected_cost=0;
        double entropy=0;
        int    min_cost=0;
        try{
            expected_cost = get_expected_cost();
            entropy = get_entropy();
            min_cost = get_minimum_cost();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        
        String sb = "";
        for (int index = 0; index < this.plan.action.length; index++) {
            if(this.plan.action[index].isBound()){
                sb+=" "+Integer.toString(plan.action[index].min())+" ";
            }
            else{
                sb+=" x ";
            }
        }


        try {
				File file = new File("./src/main/java/minicpbp/examples/data/ClassicalAIplanning/rewards.txt");
				if (!file.exists()) {
					file.createNewFile();
				}
				FileWriter fw = new FileWriter(file);
				BufferedWriter bw = new BufferedWriter(fw);

                for (int i = 0; i < this.plan.configuration_init.length; i++) {
                    bw.write(Integer.toString(this.plan.configuration_init[i]));
                    if(i<this.plan.configuration_init.length-1) bw.write(",");
                    else bw.write("\n");
                }

                for (int i = 0; i < this.plan.configuration_target.length; i++) {
                    bw.write(Integer.toString(this.plan.configuration_target[i]));
                    if(i<this.plan.configuration_target.length-1) bw.write(",");
                    else bw.write("\n");
                }
                bw.write(Integer.toString(this.plan.currentIndex+this.plan.nStays)+"\n");
                for (int i = 0; i < this.plan.currentIndex +this.plan.nStays; i++) {
                    bw.write(Integer.toString(this.plan.previousActions[i]));
                    if(i<this.plan.currentIndex-1 +this.plan.nStays) bw.write(",");
                    else bw.write("\n");
                }

				bw.write(Double.toString(expected_cost)+",");
                bw.write(Integer.toString(min_cost)+",");
                bw.write(Double.toString(this.plan.planSize));
				bw.flush();
				bw.close();
                
		file = new File("./src/main/java/minicpbp/examples/data/ClassicalAIplanning/cost_distribution.txt");
                if (!file.exists()) {
                    file.createNewFile();
                }
                fw = new FileWriter(file);
                bw = new BufferedWriter(fw);
                String distribution = "";
                for (int i = this.plan.cost.min(); i <= this.plan.cost.max(); i++) {
                    if(this.plan.cost.contains(i)){
                        distribution += Integer.toString(i)+":"+Double.toString(this.plan.cost.marginal(i));
                        bw.write(Integer.toString(i));
                        bw.write(":");
                        bw.write(Double.toString(this.plan.cost.marginal(i)));
                        if(i <= this.plan.cost.max()){ bw.write("\n"); distribution += ", ";}
                        else;// bw.write("\n");
                    }
                }
                bw.flush();
		bw.close();
                //System.out.println(distribution);
        } catch (Exception e) {
                System.out.println("write error!");
			}
         
    }
}
