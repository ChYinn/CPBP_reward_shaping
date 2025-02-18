/*
 * mini-cp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * mini-cp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2018. by Laurent Michel, Pierre Schaus, Pascal Van Hentenryck
 *
 * mini-cpbp, replacing classic propagation by belief propagation 
 * Copyright (c)  2019. by Gilles Pesant
 */

package minicpbp.examples;

import minicpbp.cp.Factory;
import minicpbp.engine.core.Constraint;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.search.LDSearch;
import minicpbp.search.Objective;
import minicpbp.search.SearchStatistics;
import minicpbp.util.io.InputReader;
import minicpbp.util.exception.InconsistencyException;
import minicpbp.util.Automaton_2D_cost;

import java.util.Arrays;

import static minicpbp.cp.BranchingScheme.*;
import static minicpbp.cp.Factory.*;

import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;


/**
 * A generic CP-BP model&solve approach to classical AI planning
 */
public class AI_Planning_Instance {

    public int minPlanLength;
    public int maxPlanLength;
    public int nbActions;
    public int objectiveCombinator; // 0 if no objective; 1/2/3 for same/sum/max
    public int lowerBoundB;
    public int lowerBoundC;
    public int nbAutomata;
    public Automaton_2D_cost[] automaton;
    public int maxActionCost;
	public int minActionCost;
    public int nbOptimizationConstraints;
    public int currentBestPlanCost;
    public int timeout = 1000; // to look for a plan of given length, in milliseconds
    public int failout = 100; // to look for a plan of given length
	
	public int planSize;
	public int currentIndex;
	public int[] previousActions; 
	public IntVar[] action;
	public IntVar cost;
	public Solver cp;
	public int bpIters = 3;
	public int nBlocks;
	public int[] configuration_init;
	public int[] configuration_target;

	public int STAY=-1;
	public int FINISH;
	public int nStays;
	public double threshold= 1.0;

	public IntVar nextVar;

	public AI_Planning_Instance(String directory_, int planSize_, int nBlocks_, int[] configuration_init_, int[] configuration_target_){
		this.nBlocks = nBlocks_;
		this.configuration_init = new int[this.nBlocks];
		for (int i = 0; i < configuration_init.length; i++) {
			this.configuration_init[i] = configuration_init_[i];
		}

		this.configuration_target = new int[this.nBlocks];
		for (int i = 0; i < configuration_target.length; i++) {
			this.configuration_target[i] = configuration_target_[i];
		}


		this.planSize =  planSize_;
		this.currentIndex = 0;
		this.previousActions = new int[planSize];
		this.nStays =0;
		create(directory_);
		boolean setFlag=true;
		for (int index = 0; index < this.action.length; index++) {
			if(!this.action[index].isBound()){
				if(setFlag){
					nextVar=this.action[index];
					setFlag=false;
				}
			}
			
		}

		//propagate();
	}
	public boolean isFinished(){
		return currentIndex==this.nbActions;
	}
        public void print_cost(){
            String elems = "<";
            for (int i = cost.min(); i <= cost.max(); i++) {
                if(cost.contains(i)){
                    elems += Integer.toString(i)+",";
                }}
            elems+=">";
            System.out.println(elems);
        }
	public void propagate(){
		this.cp.fixPoint();
		this.cp.vanillaBP(bpIters);
	}
	public void setActions(int[] action){
		//System.out.println("setting actions");
		//System.out.println(Arrays.toString(action));
		for (int i = this.currentIndex+nStays; i < Math.min(action.length, this.planSize); i++) {
			assignNext(action[i], false);
		}
	}
	public void assignNext(int a, boolean propa){
		if(a!=STAY){
			this.action[currentIndex].assign(a);
			this.previousActions[currentIndex+nStays] = a;
			currentIndex += 1;
		}
		else{
			this.action[planSize-nStays-1].assign(FINISH);
			this.previousActions[currentIndex+nStays] = a;
                        nStays += 1;
		}
		if(propa)
			propagate();
		//System.out.println("assigned"+Integer.toString(a));
		//System.out.println(Arrays.toString(previousActions));
	}
	/*
	public Double get_expected_cost(){
		double E = 0;
		double cumul=0;
		int botK = 5;
		for (int i = cost.min(); i <= Math.min(cost.max(),cost.min()+botK); i++) {
			E += cost.marginal(i)*i;
			cumul = cumul + cost.marginal(i);
		}
		return E/cumul;
	}
	 */

	private double softmax(double input, double[] neuronValues, double t) {
		double temperature = t;
		double[] temperedNeurons = new double[neuronValues.length];
		for (int i = 0; i < temperedNeurons.length; i++) {
			temperedNeurons[i] = neuronValues[i]/temperature;
		}
		double total = Arrays.stream(temperedNeurons).map(Math::exp).sum();
		return Math.exp(input/temperature) / total;
	}

	public void setFinishOracle(){
		int[] vals = new int[this.nbActions];
        for (int i = 0; i < vals.length; i++) {
            vals[i]=i;
        }
        double[] distribution_ = new double[this.nbActions];
        double norm = 0;
        for (int i = 0; i < distribution_.length; i++) {
			if(i==this.FINISH){
            	distribution_[i] = 1;
				norm+=1;
			}
			else{
				distribution_[i] = 1;
				norm+=1;
			}
        }
		for (int i = 0; i < distribution_.length; i++) {
            //distribution_[i] /= norm;
        }
		//System.out.println(Arrays.toString(distribution_));
		boolean setFlag=true;
		for (int index = 0; index < this.action.length; index++) {
			
			if(!this.action[index].isBound()){
				if(this.action[index].contains(this.FINISH)){
					Constraint o = Factory.oracle(this.action[index], vals, distribution_);
					o.setWeight(0.00000000000001);
					this.cp.post(o);
				}
			}
				
			
		}
	}

	public void setOracle(Double[] distribution){
        int[] vals = new int[distribution.length];
        for (int i = 0; i < vals.length; i++) {
            vals[i]=i;
        }
        double[] distribution_ = new double[distribution.length];
        double norm = 0;
        for (int i = 0; i < distribution_.length; i++) {
            distribution_[i] = distribution[i];
        }
   
        /*
        for (int i = 0; i < distribution_.length; i++) {
            distribution_[i] /= norm;
        }*/
        //System.out.print("\n");
        //System.out.print("neural distribution:");
        for (int i = 0; i < distribution_.length; i++) {
            if(!this.action[this.currentIndex].contains(i)){
                distribution_[i]=0;
            }
            else{
				norm+=distribution_[i];
            }
            //System.out.print(Integer.toString(i)+": ");
           // System.out.print(distribution_[i]);
            //System.out.print(", ");
        }
		for (int i = 0; i < distribution_.length; i++) {
            distribution_[i]/=norm;
			//System.out.print(distribution_[i]);
			//System.out.print(", ");
		}
		//System.out.print("\n");
		//System.out.println("norm: "+norm);

        //System.out.print("\n----------\n");
        
        //Oracle o = new Oracle(this.action[this.currentIndex],vals,distribution_);
		int next=this.action.length-1;
		for (int index = 0; index < this.action.length; index++) {
			if(!this.action[index].isBound())
				next = index;
				//break;
			else{
				Constraint o = Factory.oracle(this.action[next], vals, distribution_);
				o.setWeight(10);
				this.cp.post(o);
			}	
		}


        

        
    }
	public Double get_std(){
		double std = 0;
		double E = cost.min();
		//int rolls = 0;
		for (int i = cost.min(); i <= cost.max(); i++) {
			std += cost.marginal(i)*((E-i)*(E-i));
			//rolls+=1;
		}
		//System.out.println("std: "+Double.toString(std));
		//System.out.println("E: "+Double.toString(E));
		//System.out.println("rolls: "+Double.toString(rolls));
		return Math.sqrt(std);
	}
	/* */
	public Double get_entropy(){
		double m = this.nStays + this.currentIndex;
		double E = 0;
		double cumul=0;
		double threshold = 0.25;//Math.min(0.025+m/(2*this.maxPlanLength),0.5); //= 0.15;
		//System.out.println(threshold);
		for (int i = cost.min(); i <= cost.max(); i++) {
			if(cumul + cost.marginal(i) < threshold){
				E += cost.marginal(i)*Math.log(cost.marginal(i));
				cumul = cumul + cost.marginal(i);
			}
			else{
				E+=(threshold - cumul)*i;
				break;
			}
		}
		return E/threshold + this.nStays;
		
	}
	/*
	public Double get_expected_cost(){
		double m = this.nStays + this.currentIndex;
		double t = Math.max(((double)(this.planSize-m))/10,1);//0.5;
		double E = 0;
		int[]       vals = new int[cost.max()- cost.min()+1];
		double[] weights = new double[cost.max()- cost.min()+1];
		double[] soft_distribution = new double[cost.max()- cost.min()+1];
		int k=0;
		double offset = 0;
		for (int i = cost.min(); i <= cost.max(); i++) {
			if(cost.max()>0){
				offset = cost.max();
			}
			weights[k] = (i*-1 + offset);///nbAutomata;
			vals[k] = i;
			k+=1;
		}
		double tot_soft = 0;
		for (int i = 0; i < soft_distribution.length; i++) {
			soft_distribution[i] = softmax(weights[i],weights,t);//*cost.marginal(vals[i]);
			tot_soft+= soft_distribution[i];
		}
		for (int i = 0; i < soft_distribution.length; i++) {
			soft_distribution[i]/=tot_soft;
		}
		for (int i = 0; i < vals.length; i++) {
			E += vals[i]*soft_distribution[i];
		}
		return E + this.nStays;
	} */
	
        // with CPBP
	
	public Double get_expected_cost(){
            double E = 0;
            double cumul=0;
			double threshold = 1.0;
			if(nBlocks==2){
				threshold=1.0;
			}
            //double threshold = this.threshold;
            //System.out.println("threshold: "+threshold);
            for (int i = cost.min(); i <= cost.max(); i++) {
                    if(cumul + cost.marginal(i) < threshold){
                            E += cost.marginal(i)*i;
                            cumul = cumul + cost.marginal(i);
                    }
                    else{
                            E+=(threshold - cumul)*i;
                            break;
                    }
            }
            return E/threshold + this.nStays;
	}
	// with CPBP, weighted
	/* 
	public Double get_expected_cost(){
            double E = 0;
            double cumul=0;
			double k = 0.0;
            for (int i = cost.min(); i <= cost.max(); i++) {
				k= (i-cost.min()+0.0);
				E += 1.0/(1.0+k*k)*cost.marginal(i)*i;
				cumul = cumul + 1.0/(1.0+k*k)*cost.marginal(i);
				
            }
			System.out.println( E/cumul);
            return E/cumul + this.nStays;
	}*/
        
        // without CPBP
        /*
	public Double get_expected_cost(){
            double E = 0;
            double cumul=0;
            double threshold = 1.0;
            double m_flat  = 1.0/cost.size();
            //System.out.println("flat_marginal:"+m_flat+"--"+cost.toString());
            
            String elems = "<";
            for (int i = cost.min(); i <= cost.max(); i++) {
                if(cost.contains(i)){
                    elems += Integer.toString(i)+",";
                }}
            elems+=">";
            
            for (int i = cost.min(); i <= cost.max(); i++) {
                if(cost.contains(i)){
                    if(cumul + m_flat < threshold){
                        E += m_flat*i;
                        cumul = cumul + m_flat;
                    }
                    else{
                        E+=(threshold - cumul)*i;
                        cumul = threshold;
                        break;
                    }
                }
            }
            System.out.println("E:"+E+", E/cumul:"+E/cumul+", set:"+elems);
            return E/cumul + this.nStays;
	}*/
        
	public Double get_expected_cost_flat(){
            double E = 0;
            int m = 0;
            for (int i = cost.min(); i <= cost.max(); i++) {
                if(cost.contains(i)){
                    E += i;
                    m += 1;
                }
            }
            E /=m ;
            //E = cost.max();
            //double E_cpbp = get_expected_cost_CPBP();
            //System.out.println(E + this.nStays+" vs. "+E_cpbp);
            return E + this.nStays;
	} 
        
        /*
	public Double get_expected_cost(){
		double E = 0;
		for (int i = cost.min(); i <= cost.max(); i++) {
			E += cost.marginal(i)*i;
		}
		return E + this.nStays;
	} */
	public int get_minimum_cost(){
	    return cost.min() + this.nStays;//(cost.max() + this.nStays - (cost.min() + this.nStays))/2;
	} 
    public void print2D(int mat[][]){
        // Loop through all rows
        for (int i = 0; i < mat.length; i++){
            // Loop through all elements of current row
            for (int j = 0; j < mat[i].length; j++){
                System.out.print(mat[i][j] + " ");
            }
            System.out.print("\n");
        }
    }
    public void create(String directory) {

	//###########################################
	// read the instance, whose name is the first argument
	InputReader reader = new InputReader(directory + "problem.txt");
        int finish_int = reader.getInt();
	minPlanLength = reader.getInt();
	maxPlanLength = reader.getInt();
	nbActions = reader.getInt();
	this.FINISH = finish_int;//nbActions-1;
	this.STAY = -1;//nbActions-2;
	objectiveCombinator = reader.getInt();
	if (objectiveCombinator==0) {// no action costs
	    lowerBoundB = 0;
	    lowerBoundC = 1; // lower (and upper) bound is length
	    maxActionCost = 1;
	}
	else {
	    lowerBoundB = reader.getInt();
	    lowerBoundC = reader.getInt();
	    maxActionCost = 0;
	}
	nbAutomata = reader.getInt();
	automaton = new Automaton_2D_cost[nbAutomata];
	nbOptimizationConstraints = 0;
	for(int i=0; i<nbAutomata; i++) {
	    automaton[i] = new Automaton_2D_cost(reader,nbActions);
	    if (automaton[i].optimizationConstraint()) {
		nbOptimizationConstraints++;
		int[][] localActionCost = automaton[i].actionCost();

		for (int j = 0; j < localActionCost.length; j++) {
			for (int k = 0; k < localActionCost[0].length; k++) {
				if (localActionCost[j][k] > maxActionCost) maxActionCost = localActionCost[j][k];
				if (localActionCost[j][k] < minActionCost) minActionCost = localActionCost[j][k];
			}
		}

	    }
	}
	//###########################################
	currentBestPlanCost = maxActionCost*this.planSize*nbAutomata; //* maxActionCost + 1; // trivial strict upper bound
	//###########################################
	// try plans of increasing length
	int length = planSize+1;
        //int length = planSize;
		// define the CP model
		this.cp = makeSolver();
		// decision variables defining the sequential plan: action[0],action[1],...,action[length-1]
		this.action = new IntVar[length];
		for (int i = 0; i < length; i++) {
		    action[i] = makeIntVar(cp, 0, nbActions-1);
		}
		action[length-1].assign(finish_int);
		// objective to minimize
		//IntVar planCost = makeIntVar(cp, (0), length);
                IntVar planCost = makeIntVar(cp, 0, length+maxActionCost);
                this.cost =  planCost;
                //System.out.println("Initialized cost...");
                //this.print_cost();
                
		IntVar[] automataCosts = new IntVar[nbOptimizationConstraints];
		int k = 0;
		// for each component of factored transition system...
		for(int i=0; i<nbAutomata; i++) {
		    IntVar[] localAction = new IntVar[length];
		    // map the original actions to these local actions
		    for (int j = 0; j < length; j++) {
			localAction[j] = makeIntVar(cp, 0, automaton[i].nbLocalActions()-1);
			cp.post(table(new IntVar[]{action[j],localAction[j]},automaton[i].actionMap()));
		    }
                    
                    
                    
                    //System.out.println("Transition:");
                    //print2D(automaton[i].transitionFct());
                    //System.out.println("Accepts:");
                    //System.out.println((automaton[i].goalStates().toString()));
                    //System.out.println("Initial:"+Integer.toString(automaton[i].initialState()));
		    // post one (cost)regular constraint	
		    if (automaton[i].optimizationConstraint()) {
			if (objectiveCombinator >= 2) {
			    //IntVar automatonCost = makeIntVar(cp, 0, length);
                            //System.out.println(maxActionCost);
                            IntVar automatonCost = makeIntVar(cp, 0, length+maxActionCost);
			    automataCosts[k++] = automatonCost;
			    cp.post(costRegular(localAction, automaton[i].transitionFct(), automaton[i].initialState(), automaton[i].goalStates(), automaton[i].actionCost(), automatonCost));
			}
			else { // objectiveCombinator == 1 i.e. same
			    cp.post(costRegular(localAction, automaton[i].transitionFct(), automaton[i].initialState(), automaton[i].goalStates(), automaton[i].actionCost(), planCost));
			}
		    } else
			cp.post(regular(localAction, automaton[i].transitionFct(), automaton[i].initialState(), automaton[i].goalStates()));
                    //System.out.println("Automaton "+i+"...");
                    //this.print_cost();
		}
		// express planCost as combination of automataCost
		switch(objectiveCombinator) {
		case 0: // no objective
		    planCost.assign(length);
		    break;
		case 1: // same; already taken care of
		    break;
		case 2: // sum
		    cp.post(sum(automataCosts,planCost));
		    break;
		case 3: // max
                    //System.out.println("Setting cost ...");
                    //this.print_cost();
		    cp.post(maximum(automataCosts,planCost));
                    //this.print_cost();
		    break;
		}
                
		//this.cost =  planCost;
                //System.out.println("Exiting cost...");
                //this.print_cost();
    }
}
