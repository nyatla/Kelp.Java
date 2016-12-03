package jp.nyatla.kelpjava.test;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import jp.nyatla.kelpjava.common.NdArray;

public class SerialieTest {
	public static void main(String[] args){
        // シリアライズ
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("person.txt"))) {
    		NdArray na=new NdArray(new double[]{0,1,2,3},new int[]{1},false);
            oos.writeObject(na);
        } catch (IOException e) {
        	e.printStackTrace();
        }

	}
}
