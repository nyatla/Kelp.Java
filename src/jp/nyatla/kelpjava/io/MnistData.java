package jp.nyatla.kelpjava.io;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import jp.nyatla.kelpjava.common.Mother;
import jp.nyatla.kelpjava.common.NdArray;

/**
 * Mnistデータを格納するクラスです。
 * Mnist形式の画像とNdArrayでのアクセス手段を提供します。
 * 
 */
public class MnistData {
	public class DataSet
	{
		final public NdArray[] image;
		final public NdArray[] label;

		protected DataSet(NdArray[] i_image, NdArray[] i_teach) {
			this.image = i_image;
			this.label = i_teach;
		}
	}

	final private List<NdArray> image = new ArrayList<NdArray>();
	final private List<NdArray> label = new ArrayList<NdArray>();

	public MnistData(File i_image_file,File i_label_file) throws IOException {
		this(new MnistImageFile(i_image_file),new MnistLabelFile(i_label_file));
	}

	public MnistData(MnistImageFile i_image_file,MnistLabelFile i_label_file) {
		for (int i = 0; i < i_image_file.numberOfImages; i++) {
			// トレーニングデータ
			double[] data = new double[28 * 28];
			for (int p = 0; p < 28 * 28; p++) {
				data[p] = (i_image_file.bitmapList[i][p] & 0x000000ff)/255.0;
			}
			NdArray img = new NdArray(data, new int[] { 1, 28, 28 }, false);
			// トレーニングデータラベル
			double[] teach = new double[1];
			teach[0] = (int) i_label_file.labelList[i];
			NdArray label = new NdArray(teach, new int[] { 1 }, false);
			this.image.add(img);
			this.label.add(label);
		}
	}

	/**
	 * ランダムに選択したデータセットを返します。
	 * @param i_dataCount
	 * データセットの数を指定します。
	 * @return
	 */
	public DataSet getRandomDataSet(int i_dataCount)
	{
		int s = this.image.size();
		NdArray[] image=new NdArray[i_dataCount];
		NdArray[] label=new NdArray[i_dataCount];
		for (int j = 0; j < i_dataCount; j++) {
			int index = Mother.Dice.nextInt(s);
			image[j]=this.image.get(index);
			label[j]=this.label.get(index);
		}
		return new DataSet(image,label);
	}
}
