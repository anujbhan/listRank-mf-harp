package soic.iu.edu;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;


public class ListRankMFSequential {
	
	//user_item_rating_map
	Map<String,Map<String,Double>> r;
	Set<String> users, items;
	
	Map<String,Double[]>U_prime;
	Map<String,Double[]>V_prime;
	Map<String,Double[]>U;
	Map<String,Double[]>V;
	
	//The regularization coefficient lambda
	Double lambda = 0.01;
	// epsilon
	Double eps = 0.01;
	//Number of iterations
	int iterations = 10;
	//learning rate
	Double lr = 0.001;
	//Number of features
	int F = 5;
	public int getF(){return F;}
	
	//Input data conversion (get items)
	public static Set<String> r_u_i_TO_i(Map<String,Map<String,Double>> r) {
		Set<String> items = new HashSet<String>();
		
		for(String u : r.keySet())
			for(String i : r.get(u).keySet()) 
				items.add(i);
		
		return items;
	}	

	//Input data conversion (get users)
	public static Set<String> r_u_i_TO_u(Map<String,Map<String,Double>> r) {
		return r.keySet();
	}	

	
	//Input data format:
	//User Item Rating
	//Example:
	//U9823 I9823 5.0
	public static Map<String,Map<String,Double>> readData(String filename) throws Exception {
		Map<String,Map<String,Double>> r = new HashMap<String,Map<String,Double>>();
		
		System.out.println("Reading file " + filename + " ...");
		BufferedReader br = new BufferedReader( new FileReader(filename) );
		String line;
		while ( (line = br.readLine()) != null )  {
			//System.out.println("Reading line: " + line);
			String[] array = line.split("\t");
			if (array.length == 1){
				array = line.split(" ");
				if (array.length == 1)
					continue;
			}
			String user = array[0];
			String item = array[1];
			Double rating = Double.parseDouble(array[2]);
			if( !r.containsKey(user) ) r.put(user, new HashMap<String,Double>());
			
			r.get(user).put(item,rating);
		}
		
		System.out.println("End of reading file " + filename);
		
		return r;
	}

	//Logistic function
	public static Double logisticFunction(Double x){
		return(1/(1 + Math.exp(-x)));
	}
	//The d/dx of the logistic function
	public static Double logisticFunctionDerivative(Double x){
		Double y = logisticFunction(x);
		return y * (1-y);
	}
	
	// returns the norm of a vector^2
	public Double vectorNormSqrd(Double[] x){
		Double s = 0.0;
		for(int i=0;i<x.length;i++){
			s += Math.pow(x[i], 2);
		}
		return s;
	}
			
	// returns the rating of item i for user u
	// returns 0.0 if rating is not found
	public Double getRating(String u, String i){
		Double rating = 0.0;
		if(r.containsValue(u)){
			if(r.get(u).containsKey(i)){
				rating = r.get(u).get(i);
			}
		}
		return rating;
	}
	
	// prediction is the result of vector product of user's features by item's feature
	public Double predictRating(Double[] u, Double[] v){
		Double uv = 0.0;
		for(int f=0; f<u.length; f++){ 
			uv += u[f]*v[f];
		}

		return uv;
	}
	
	// The loss function as described in the ListRank-MF paper
	public Double L(Map<String,Double[]> U, Map<String,Double[]> V){
		Double loss =0.0;
			
		for(String user: r.keySet()){
			Double denom_rating = 0.0;
			Double denom_prediction = 0.0;
			
			for(String item: r.get(user).keySet()){
				denom_rating += Math.exp(getRating(user,item));
				denom_prediction += Math.exp(logisticFunction(predictRating(U.get(user),V.get(item))));
			}
			
			for(String item: r.get(user).keySet()){
				loss -= Math.exp(getRating(user,item))/denom_rating * Math.log(Math.exp(logisticFunction(predictRating(U.get(user),V.get(item)))/denom_prediction));
			}
		}
		
		//note: the Frobenius norm in the paper can also be considered as the vector norm. 
		for(String user: U.keySet()){
			loss += (lambda/2)*vectorNormSqrd(U.get(user));
		}
		for(String item: V.keySet()){
			loss += (lambda/2)*vectorNormSqrd(V.get(item));
		}
		
		return loss;
	}

	//gradient Ui
	public void computeUi(Map<String,Double[]>U, Map<String,Double[]>V, String user){
		
		
		Double denom_prediction = 0.0;
		Double denom_rating = 0.0;
		
		for(String item: r.get(user).keySet()){
			
			denom_prediction += Math.exp(logisticFunction(predictRating(U.get(user), V.get(item))));
			denom_rating += Math.exp(getRating(user,item));
			
		}
			
		
		for(String item: r.get(user).keySet()){
			Double rhat = predictRating(U.get(user), V.get(item));
			
			Double value = logisticFunctionDerivative(rhat);
			value *= ( Math.exp(logisticFunction(rhat)) / denom_prediction ) - 
					 ( Math.exp(getRating(user,item))   / denom_rating );
		
			for(int i =0;i<getF();i++){
				U_prime.get(user)[i] += value * V.get(item)[i];
			}
			
		}
		
		for(int i =0;i<getF();i++){
			U_prime.get(user)[i] += lambda * U.get(user)[i];
		}
		
	}

	//gradient Vj
	public void computeVj(Map<String,Double[]>U, Map<String,Double[]>V, String item){
		
		for(String user: r.keySet()){
			
			Double denom_prediction = 0.0;
			Double denom_rating = 0.0;
			
			for(String i : r.get(user).keySet()){
				denom_prediction += Math.exp(logisticFunction(predictRating(U.get(user), V.get(i))));
				denom_rating += Math.exp(getRating(user,i));				
			}
			
			Double rhat = predictRating(U.get(user),V.get(item));
			Double value = logisticFunctionDerivative(rhat);
			value *= ( Math.exp(logisticFunction(rhat) / denom_prediction) ) -
					 ( Math.exp(getRating(user,item)) /  denom_rating);
		
			for(int i =0;i<getF();i++){
				V_prime.get(item)[i] += value * U.get(user)[i];
			}
		
		}
		
		for(int i =0;i<getF();i++){
			V_prime.get(item)[i] += lambda * V.get(item)[i];
		}
		
	}

	public ListRankMFSequential(Map<String,Map<String,Double>> r){
		this.r = r;
		this.items = r_u_i_TO_i(r);
		this.users = r_u_i_TO_u(r);

		// init
		System.out.println("Initilizing...");

		U_prime = new HashMap<String,Double[]>();
		V_prime = new HashMap<String,Double[]>();
		U = new HashMap<String,Double[]>();
		V = new HashMap<String,Double[]>();
		
		// init U_prime
		for(String u : users) {
			Double[] vec = new Double[F];
			for(int f=0; f<getF(); f++) 
				vec[f] = 0.0;
			U_prime.put(u, vec);
		}
		System.out.println("Done initilizing U_prime.");

		//init V_prime
		for(String i : items) {
			Double[] vec = new Double[F];
			for(int f=0; f<getF(); f++) 
				vec[f] = 0.0;
			V_prime.put(i, vec);
		}
		System.out.println("Done initilizing V_prime.");
		
		// init U
		for(String u:users){
			Double [] rand = new Double[getF()];
			for(int j=0;j<rand.length;j++){
				Random rndm = new Random();
				rand[j] =  rndm.nextDouble();;
			}
			U.put(u, rand);
		}
		System.out.println("Done initilizing U.");

		// init V
		for(String i:items){
			Double [] rand = new Double[getF()];
			for(int j=0;j<rand.length;j++){
				Random rndm = new Random();
				rand[j] =  rndm.nextDouble();
			}
			V.put(i, rand);
		}
		System.out.println("Done initilizing V.");

		Double lloss = Double.MAX_VALUE;
		
		// SGD
		System.out.println("Running Stochastic Gradient Descent (SGC) iterations...");

		for(int iter=0; iter < iterations ; iter++){
			System.out.println("Iterations : " + (iter+1));
            System.out.println("calc loss start ..");
            Double loss = L(U,V);
            System.out.println("\t\t Loss: "+loss);
			if (loss > lloss - eps) break;
			
			lloss = loss;
			
			// best U
			for(String user : users){
				computeUi(U,V,user);
				
				for(int j = 0; j<getF();j++){
					U.get(user)[j] -= lr * U_prime.get(user)[j];
				}
			}
            System.out.println("computeUi complete ...");
            // best V
			for(String item : items){
				computeVj(U,V,item);
				
				for(int j = 0; j<getF();j++){
					V.get(item)[j] -= lr * V_prime.get(item)[j];
				}
			}
            System.out.println("computeVj complete...");
        }
	}
	
	public Double finalPrediction(String user, String item){
		return predictRating(U.get(user), V.get(item));
	}
	
	public static void main(String[] args) throws Exception {
		String inputFile = System.getProperty("user.dir") + File.separator +
				"src" + File.separator + "main" + File.separator +
				"resources" + File.separator + "sample.txt";
		NetflixDataCleaner netflixDataCleaner = new NetflixDataCleaner();
		Map<String,Map<String,Double>> r = netflixDataCleaner.dataExtractor("/mnt/dataDrive/download/data");
		ListRankMFSequential lrmf = new ListRankMFSequential(r);
		System.out.println("\t\tU1-1488844: "+ lrmf.finalPrediction("1488844","3"));
		System.out.println("\t\tU1-1695221: "+ lrmf.finalPrediction("1695221","5"));
		//System.out.println("\t\tU1-2649429: "+ lrmf.finalPrediction("2649428","4"));
		//System.out.println("\t\tU1-1234234: "+ lrmf.finalPrediction("1234234","5"));
//		Map<String,Map<String,Double>> r = readData(inputFile);
//		ListRankMFSequential lrmf = new ListRankMFSequential(r);
//		System.out.println("\t\tU1-2: "+ lrmf.finalPrediction("U1","2"));
//		System.out.println("\t\tU1-3: "+ lrmf.finalPrediction("U1","3"));
//		System.out.println("\t\tU1-4: "+ lrmf.finalPrediction("U1","4"));
//		System.out.println("\t\tU1-5: "+ lrmf.finalPrediction("U1","5"));
	}

}
