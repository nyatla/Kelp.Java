package jp.nyatla.kelpjava.common;

public class JavaUtils {
	public static double[] cloneArray(double[] i_source){
		return i_source.clone();
	}
	public static double max(double[] i_v)
	{
		double max=Double.MIN_VALUE;
		for(int i=i_v.length-1;i>=0;i--){
			if(max<i_v[i]){
				max=i_v[i];
			}
		}
		return max;
	}
	public static double sum(double[] i_v)
	{
		double sum=0;
		for(int i=i_v.length-1;i>=0;i--){
			sum+=i_v[i];
		}
		return sum;
	}
	public static int indexOf(double[] i_array,double i_v)
	{
		for(int i=i_array.length-1;i>=0;i--){
			if(i_v==i_array[i]){
				return i;
			}
		}
		return -1;
	}
	
}
