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

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;

public class LabelBinarizer {
    Map<String, Integer> labelToIdx = new HashMap<String, Integer>();
	
	public LabelBinarizer(){}
	
	public void setLabelToIdx(Map<String, Integer> labelToIdx){
		this.labelToIdx = labelToIdx;
	}
	
    public void fit(String[] labels) {
		String[] uniqueLabels = Arrays.stream(labels).distinct().toArray(String[]::new);
        int index = 1;
        for (String label : uniqueLabels) labelToIdx.put(label, index++);
    }

	public int[] transformDense(String[] labels) {
        int[] binarized = new int [labels.length];
        for (int i = 0; i < labels.length; i++) {
            Integer index = labelToIdx.get(labels[i]);
            if (index != null) {
                binarized[i] = index;
            }else binarized[i] = -1;//not found
        }
		return binarized;
    }

	public String[] getLabels(String colName){
		List<String> labels = new ArrayList<String>(labelToIdx.keySet());
		String[] res = new String [labels.size()];
		for (int i=0;i<labels.size();i++) res[i] = colName + "=" +labels.get(i);
		return res;
	}
}
