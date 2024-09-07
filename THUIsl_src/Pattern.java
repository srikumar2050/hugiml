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

import java.util.ArrayList;
import java.util.List;

public class Pattern implements Comparable<Pattern>{
	
	final String prefix;
	final float utility;
	final int sup;
	final int idx;//for sorting patterns in order of insertion 
	List<Integer> tidList = new ArrayList<Integer>();
	double entropy = 0;
	double ig = 0;//information gain
	int pureClass = -1;
	
	public Pattern(int[] prefix, int length, UtilityList X, int idx) {
		StringBuilder buffer = new StringBuilder();
		for (int i=0;i<length; i++){
			buffer.append(prefix[i]);
			buffer.append(" ");
		}
		buffer.append(X.item);
		this.prefix = buffer.toString();
		this.idx = idx;
		
		this.utility = X.getUtils();
		this.sup = X.elements.size();
		
		for (Element e : X.elements){
			this.tidList.add(e.tid + 1);
		}
		//this.patternLength = length + 1;
		this.entropy = X.entropy;
		this.ig = X.ig;
		
		if (this.entropy==0){
			this.pureClass = X.pureClass;
		}
	}
	
	public Pattern(String p, int support){
		this.prefix = p;
		this.utility = 0;
		this.sup = support;
		this.tidList = null;
		this.idx = 0;
	}

	public String getPrefix(){
		return this.prefix;
	}

	public int compareTo(Pattern o) {
		if(o == this){
			return 0;
		}
		float compare = this.utility - o.utility;
		if(compare !=0){
			return (int) compare;
		}
		return this.hashCode() - o.hashCode();
	}

}
