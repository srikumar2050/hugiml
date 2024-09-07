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
import java.lang.Math;

class UtilityList {
	final int item;  // the item
	float sumIutils = 0;  //the sum of item utilities
	float sumRutils = 0;  //the sum of remaining utilities
	List<Element> elements = new ArrayList<Element>();  //the elements
	List<Integer> ytrainCurr = new ArrayList<Integer>();//ytrain - to calculate entropy of each pattern
	int[] classCounts;
	double entropy = 0;
	double ig = 0;//information gain 
	int pureClass = -1;
	
	public UtilityList(int item){
		this.item = item;
	}	
	
	public float getUtils(){
		return this.sumIutils;
	}
	
	public void addElement(Element element){
		sumIutils += element.iutils;
		sumRutils += element.rutils;
		elements.add(element);
	}
	
	public double getError(List<Double> ytrain){
		double ybar = 0.0, sqdiff = 0.0, error = 0.0;
		double diff = 0.0;
		for (Double y:ytrain) ybar += y;
		ybar = ybar/ytrain.size(); 
		
		for (Double y:ytrain) diff += Math.abs((y - ybar)/y);
		double mape = diff/ytrain.size();
		return mape;
	}
	
	public void setStats(UtilityList parent, List<Integer> ytrain, int num_classes){
		this.classCounts = new int [num_classes];
		for (int ci=0;ci<num_classes; ci++) this.classCounts[ci] = 0;

		for (Element e : this.elements){
			this.ytrainCurr.add(ytrain.get(e.tid));
			this.classCounts[ytrain.get(e.tid)] += 1;//set only for specific transactions where the itemset appears
		}
		this.entropy = getEntropy(this.classCounts);
		this.ig = computeIG(parent, ytrain, num_classes);
		
		if (this.entropy==0){
			for (int ci=0;ci<num_classes; ci++){
				if (this.classCounts[ci]>0){//rest of classes will have zero values for zero entropy case
					this.pureClass = ci;
					break;
				}
			}
		}	
	}
	
	public float[] getProportion(int[] classCounts){
		float[] prop = new float [classCounts.length];//proportion of each class 
		int tot = 0;
		for (int i=0;i<classCounts.length;i++) tot += classCounts[i];
		for (int i=0;i<prop.length;i++) prop[i] = (float)classCounts[i]/tot;
		
		return prop;
	}
	
	public double getEntropy(int[] classCounts){
		float[] prop = getProportion(classCounts);//proportion of each class 
		double ent = 0;
		for (int i=0;i<prop.length;i++)
			if (prop[i]!=0)
				ent += (prop[i]*Math.log(prop[i]));
		if (ent==0) return ent;
		
		return -ent;
	}
	
	public double getCrossEntropy(int[] qCounts, int[] pCounts){
		double crossEnt = 0;
		float[] q = new float [qCounts.length];
		float[] p = new float [pCounts.length];
		int totQ = 0, totP = 0;
		for (int i=0;i<qCounts.length;i++) totQ += qCounts[i];
		for (int i=0;i<qCounts.length;i++) q[i] = (float)qCounts[i]/totQ;
		for (int i=0;i<pCounts.length;i++) totP += pCounts[i];
		for (int i=0;i<pCounts.length;i++) p[i] = (float)pCounts[i]/totP;
		
		for (int i=0;i<q.length;i++)
			if (q[i]!=0)
				crossEnt += (p[i]*Math.log(q[i]));
		if (crossEnt==0) return crossEnt;
		
		return -crossEnt;
	}
	
	public int[] getOverallClassCounts(List<Integer> ytrain, int num_classes){
		int[] tmpclasscounts = new int [num_classes];
		for (int ci=0;ci<num_classes; ci++) tmpclasscounts[ci] = 0;
		for (Integer yy : ytrain) tmpclasscounts[yy] += 1;
		return tmpclasscounts;
	}
	
	public double computeIG(UtilityList parent, List<Integer> ytrain, int num_classes){
		double baseEntropy = 1;
		int[] pClassCounts = new int [num_classes];
		if (parent==null){
			pClassCounts = getOverallClassCounts(ytrain, num_classes);
			baseEntropy = getEntropy(pClassCounts);
		}else baseEntropy = getEntropy(parent.classCounts);
		
		double currEntropyLeft = this.entropy;
		int[] remainingClassCounts;
		if (parent==null){
			remainingClassCounts = new int [pClassCounts.length];
			for (int i=0;i<remainingClassCounts.length;i++) 
				remainingClassCounts[i] = pClassCounts[i] - this.classCounts[i];
		}else{
			remainingClassCounts = new int [parent.classCounts.length];
			for (int i=0;i<remainingClassCounts.length;i++) 
				remainingClassCounts[i] = parent.classCounts[i] - this.classCounts[i];
		}
		double currentEntropyRight = getEntropy(remainingClassCounts);
		
		int totalParent = 0, totalLeft = 0;
		float propLeft = 0, propRight = 0;
		if (parent==null){
			for (int i=0;i<pClassCounts.length;i++) totalParent += pClassCounts[i];
		}else{
			for (int i=0;i<parent.classCounts.length;i++) totalParent += parent.classCounts[i];
		}
		for (int i=0;i<this.classCounts.length;i++) totalLeft += this.classCounts[i];
		propLeft = (float)totalLeft/totalParent;
		propRight = 1 - propLeft;
		
		double ig = baseEntropy - propLeft*currEntropyLeft - propRight*currentEntropyRight;
		return ig;
	}

	public boolean tidsMatch(List<Element> match){
		if (match.size()!=this.elements.size()) return false;
		for (int i=0;i<this.elements.size();i++)
			if (this.elements.get(i).tid != match.get(i).tid) return false;//match every tid element
		return true; //all tid elements match
	}
}
