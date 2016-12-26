package jp.nyatla.kelpjava.io.mnist;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import jp.nyatla.kelpjava.j2se.BinaryReader;

/**
 * MNIST の画像をロードするためのクラス. http://yann.lecun.com/exdb/mnist/
 * 
 */
public class MnistImageFile {
	/**
	 * 0x0000 から始まるマジックナンバー. 0x00000803 (2051) が入る.
	 */
	final public int magicNumber;

	/**
	 * 画像の数.
	 */
	final public int numberOfImages;

	/**
	 * 画像の縦方向のサイズ.
	 */
	final public int numberOfRows;

	/**
	 * 画像の横方向のサイズ.
	 */
	final public int numberOfColumns;

	/**
	 * 画像の配列. Bitmap 形式で取得する場合は GetBitmap(index) を使用する.
	 */
	final public byte[][] bitmapList;

	public MnistImageFile(File i_file) throws IOException {
		this(new FileInputStream(i_file));
	}

	public MnistImageFile(InputStream i_stream) throws IOException
	{
		BinaryReader br = new BinaryReader(i_stream, BinaryReader.ENDIAN_BIG);
		this.magicNumber = br.getInt();
		this.numberOfImages = br.getInt();
		this.numberOfRows = br.getInt();
		this.numberOfColumns = br.getInt();

		int pixelCount = this.numberOfRows * this.numberOfColumns;
		this.bitmapList = new byte[this.numberOfImages][];
		for (int i = 0; i < this.numberOfImages; i++) {
			this.bitmapList[i] = br.getByteArray(pixelCount);
		}
	}
}
