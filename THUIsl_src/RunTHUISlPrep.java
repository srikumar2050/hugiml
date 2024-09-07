/* This file is copyright (c) 2024 Srikumar Krishnamoorthy
 * 
 * This program is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class RunTHUISlPrep {
	public static void main(String [] args) throws IOException{
		String input = "";
        String output = "";
        int topK = 100;
        int B = -1;
        int fsK = topK;
        boolean eucsPrune = false;
        String dsName = null;
        int foldNo = 1;
        int L = 1;//default value
        double G = 1e-4;//default value
		double imbWeights = 1.0;
        boolean modelTest = false;
		int numRows = 100;
		int numIntCols = 0, numFloatCols = 0, numCatCols = 0; 
		int numClasses = 2;
        Map<String, String> params = new HashMap<>();
		for (String arg : args) {
            if (arg.contains("=")) {
                String[] keyValue = arg.split("=", 2);
                params.put(keyValue[0], keyValue[1]);
            }
        }
		//System.out.println(" keys received "+params.toString());
        if (params.containsKey("input")) input = ".//" + params.get("input");
		else input = ".//outputs//inpdata//";
        if (params.containsKey("output")) output = ".//" + params.get("output");
		else output = ".//outputs//hui//";
        if (params.containsKey("topK")) topK = Integer.parseInt(params.get("topK"));
        if (params.containsKey("fsK")) fsK = Integer.parseInt(params.get("fsK"));
        if (params.containsKey("B")) B = Integer.parseInt(params.get("B"));
        if (params.containsKey("eucsprune")) eucsPrune = Boolean.parseBoolean(params.get("eucsprune"));
        if (params.containsKey("dsname")) dsName = params.get("dsname");
        if (params.containsKey("foldno")) foldNo = Integer.parseInt(params.get("foldno"));
        if (params.containsKey("L")){
			if (params.get("L").equals("all")) L = -1;//mine all pattern lengths 
			else L = Integer.parseInt(params.get("L"));//mine max specified length of patterns 
		}
        if (params.containsKey("G")) G = Double.parseDouble(params.get("G"));
		if (params.containsKey("imbWeights")) imbWeights = Double.parseDouble(params.get("imbWeights"));
		if (params.containsKey("numRows")) numRows = Integer.parseInt(params.get("numRows"));
        if (params.containsKey("modeltest")) modelTest = Boolean.parseBoolean(params.get("modeltest"));
		
		if (params.containsKey("numIntCols")) numIntCols = Integer.parseInt(params.get("numIntCols"));
		if (params.containsKey("numFloatCols")) numFloatCols = Integer.parseInt(params.get("numFloatCols"));
		if (params.containsKey("numCatCols")) numCatCols = Integer.parseInt(params.get("numCatCols"));
		if (params.containsKey("numClasses")) numClasses = Integer.parseInt(params.get("numClasses"));

		AlgoTHUIsl topkalgo = new AlgoTHUIsl(topK);
		if (!modelTest){//training phase
			topkalgo.runAlgorithm(input, output, fsK, B, L, G, dsName, foldNo, imbWeights, numRows, numIntCols, numFloatCols, numCatCols, numClasses);
		}else{//test phase
			topkalgo.applyPatterns(input, output, dsName, foldNo, numRows, numIntCols, numFloatCols, numCatCols);
		}
	}
}

