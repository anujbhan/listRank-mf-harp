package soic.iu.edu;

import java.io.*;
import java.util.*;

/**
 * Created by abhandar on 11/6/16.
 */
public class NetflixDataCleaner {

    public ArrayList<String> listFilesForFolder(final File folder) {
        ArrayList<String> fileList = new ArrayList<String>();
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                fileList.add(folder+"/"+fileEntry.getName());
            }
        }
        return fileList;
    }

    public HashMap<String,Map<String, Double>> dataExtractor(String trainingSet){
        try{
            ArrayList<String> fileList = listFilesForFolder(new File(trainingSet));
            HashMap<String,Map<String,Double>> r = new HashMap<String,Map<String, Double>>();
            for(String fileName : fileList){
                BufferedReader br = new BufferedReader(new FileReader(fileName));
                String[] movieID = br.readLine().split(":");
                String line;
                System.out.println("reading movieID: " + movieID[0]);
                while ( (line = br.readLine()) != null )  {
                    String[] array = line.split(",");
                    if(array.length != 3){
                        break;
                    }
                    if( !r.containsKey(array[0]) ) r.put(array[0], new HashMap<String,Double>());
                    r.get(array[0]).put(movieID[0],Double.parseDouble(array[1]));

                }
            }
            return r;
        }catch (NullPointerException e){
            System.out.printf(e.getMessage());
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

//    public static void main(String[] args){
//        NetflixDataCleaner netflixDataCleaner = new NetflixDataCleaner();
//        String trainingSet = "/mnt/dataDrive/download/training_set";
//        netflixDataCleaner.dataExtractor(trainingSet);
//    }
}
