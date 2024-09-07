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

public class MinMaxScaler {

    public float dataMin;
    public float dataMax;

    public MinMaxScaler() {}
	public MinMaxScaler(float[] params){
		this.dataMin = params[0];
		this.dataMax = params[1];
	}

    public void fit(float[] X) {
        dataMin = Float.POSITIVE_INFINITY;
        dataMax = Float.NEGATIVE_INFINITY;
        for (float value : X) {
            if (value < dataMin) dataMin = value;
            if (value > dataMax) dataMax = value;
        }
		if (dataMax == dataMin) dataMax = dataMin + 1.0f;
    }

    public void transform(float[] X) {//updates in-place
        float range = dataMax - dataMin;
        for (int i = 0; i < X.length; i++) {
            X[i] = (X[i] - dataMin) / range;  
        }
    }

    public void fitTransform(float[] X) {
        fit(X);
        transform(X); 
    }

    public float inverseTransform(float X_scaled) {
		return (X_scaled * (dataMax - dataMin)) + dataMin;
    }

}
