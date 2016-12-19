package jp.nyatla.kelpjava.common;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;

public class JavaUtils {
	/**
	 * 配列から最大値を返します。
	 * 
	 * @param i_v
	 * @return
	 */
	public static double max(double[] i_v) {
		double max = Double.MIN_VALUE;
		for (int i = i_v.length - 1; i >= 0; i--) {
			if (max < i_v[i]) {
				max = i_v[i];
			}
		}
		return max;
	}

	/**
	 * 配列の数値を全て加算して返します。
	 * 
	 * @param i_v
	 * @return
	 */
	public static double sum(double[] i_v) {
		return sum(i_v,i_v.length);
	}
	public static double sum(double[] i_v,int i_len) {
		double sum = 0;
		for (int i = i_len - 1; i >= 0; i--) {
			sum += i_v[i];
		}
		return sum;
	}
	/**
	 * 配列の値すべてを加算して、平均値を返します。
	 * 
	 * @param i_v
	 * @return
	 */
	public static double average(double[] i_v) {
		return average(i_v,i_v.length);
	}
	public static double average(double[] i_v,int i_len) {
		return sum(i_v,i_len) / i_len;
	}
	/**
	 * 配列から値の一致する要素のインデクスを返します。
	 * 
	 * @param i_array
	 * @param i_v
	 * @return
	 */
	public static int indexOf(double[] i_array, double i_v) {
		for (int i = i_array.length - 1; i >= 0; i--) {
			if (i_v == i_array[i]) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * オブジェクトをファイルに書き出します。
	 * 
	 * @param i_object
	 * @param i_filepath
	 * @throws IOException
	 */
	public static void writeToFile(Serializable i_object, String i_filepath)
			throws IOException {
		writeToOutputStream(i_object, new FileOutputStream(i_filepath));
	}

	/**
	 * オブジェクトをファイルに書き出します。
	 * 
	 * @param i_object
	 * @param i_file
	 * @throws IOException
	 */
	public static void writeToFile(Serializable i_object, File i_file)
			throws IOException {
		writeToOutputStream(i_object, new FileOutputStream(i_file));
	}

	/**
	 * オブジェクトをストリームに書き出します。
	 * 
	 * @param i_object
	 * @param i_stream
	 * @throws IOException
	 */
	public static void writeToOutputStream(Serializable i_object,
			OutputStream i_stream) throws IOException {
		ObjectOutputStream oos = new ObjectOutputStream(i_stream);
		oos.writeObject(i_object);
	}

	/**
	 * ストリームからオブジェクトを読み出す。
	 * 
	 * @param i_stream
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	@SuppressWarnings("unchecked")
	public static <U extends Serializable> U readFromInputStream(
			InputStream i_stream) throws IOException, ClassNotFoundException {
		ObjectInputStream oos = new ObjectInputStream(i_stream);
		return (U) oos.readObject();
	}

	/**
	 * ファイルからオブジェクトを読み出す。
	 * 
	 * @param i_file
	 * @return
	 * @throws ClassNotFoundException
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static <U extends Serializable> U readFromFile(File i_file)
			throws ClassNotFoundException, FileNotFoundException, IOException {
		return readFromInputStream(new FileInputStream(i_file));
	}

	/**
	 * ファイルからオブジェクトを読み出す。
	 * 
	 * @param i_filepath
	 * @return
	 * @throws ClassNotFoundException
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static <U extends Serializable> U readFromFile(String i_filepath)
			throws ClassNotFoundException, FileNotFoundException, IOException {
		return readFromFile(new File(i_filepath));
	}

	/**
	 * 2次元配列からRank1のNdArray配列を生成します。
	 * 
	 * @param i_array
	 * @return
	 */
	public static NdArray[] createNdArray(double[][] i_array) {
		NdArray[] ret = new NdArray[i_array.length];
		for (int i = 0; i < i_array.length; i++) {
			ret[i] = new NdArray(i_array[i], true);
		}
		return ret;
	}
	public static double[] fill(double[] i_array,double i_fill_value)
	{
		for(int i=i_array.length-1;i>=0;i--){
			i_array[i]=i_fill_value;
		}
		return i_array;
	}

	public static double[][] cloneArray(double[][] xhat)
	{
		double[][] r=new double[xhat.length][];
		for(int i=0;i<xhat.length;i++){
			r[i]=new double[xhat[i].length];
			for(int j=0;j<xhat[i].length;j++){
				r[i][j]=xhat[i][j];
			}
		}
		return r;
	}
}
