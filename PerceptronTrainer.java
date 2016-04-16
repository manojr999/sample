package classifier;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.text.DecimalFormat;

public class PerceptronTrainer {
	
	private static final int NUMBER_OF_RECORDS = 100;
	private static final int NUMBER_OF_ATTRIBUTES = 4;
	
	public double[] train(String trainingDataFilePath, String trainingDataFileName){
		
		double[][] inputs = new double[NUMBER_OF_RECORDS][NUMBER_OF_ATTRIBUTES];
		int[] y = new int[NUMBER_OF_RECORDS];
		double[] weights = new double[NUMBER_OF_ATTRIBUTES + 1];
		double localError, globalError;
		int output;
		
		PerceptronPredictor perceptronPredictor = new PerceptronPredictor();

		for (int i = 0; i < weights.length; i++) {
			weights[i] = randomNumber(0,1);
		}
		
		try {
			
            BufferedReader reader = 
                Files.newBufferedReader(
                    FileSystems.getDefault().getPath(trainingDataFilePath, trainingDataFileName), 
                    Charset.defaultCharset() );
 
            
            String line = null;
            int rowNum = 0;
            
            while ((line = reader.readLine()) != null)
            {            	
            	
            	
            	String[] words = line.split(",");
            	
            	for (int i = 0; i < words.length - 1; i++) {
            		inputs[rowNum][i] = Double.valueOf(words[i]);					
				}
            	
            	y[rowNum] = Integer.valueOf(words[words.length - 1]);
            	
            	rowNum++;
            }
            
            /*for (double[] ds : inputs) {
            	
            	System.out.println("");
				
            	for (double d : ds) {
					
            		System.out.print(d + ",");
            		
				}
			}*/
            
            int theta = 0;
            double LEARNING_RATE = 0.1;
            int MAX_ITER = 1000;
            int iteration = 0;
            
    		do {
    			iteration++;
    			globalError = 0;
    			
    			//loop through all instances (complete one epoch)
    			for (int p = 0; p < NUMBER_OF_RECORDS; p++) {
    				
    				// calculate predicted class
    				output = perceptronPredictor.calculateOutput(theta, weights, inputs[p]);
    				
    				// difference between predicted and actual class values
    				localError = y[p] - output;
    				
    				//update weights and bias
    				weights[0] += LEARNING_RATE * localError;
    				
    				for (int i = 0; i < inputs[p].length; i++) {
    					
    					weights[i+1] += LEARNING_RATE * localError * inputs[p][i];
    				}
    				
    				//summation of squared error (error value for all instances)
    				globalError += (localError*localError);
    			}

    			
    		} while (globalError != 0 && iteration<=MAX_ITER);
    		
		} 
        catch (IOException ioe) {
            ioe.printStackTrace();
        }
		
		return weights;
	}
	
	/**
	 * @param min
	 * @param max
	 * @return
	 */
	private static double randomNumber(int min , int max) {
		DecimalFormat df = new DecimalFormat("#.####");
		double d = min + Math.random() * (max - min);
		String s = df.format(d);
		double x = Double.parseDouble(s);
		return x;
	}
	
}
