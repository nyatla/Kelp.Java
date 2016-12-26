package jp.nyatla.kelpjava.io.mnist;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import jp.nyatla.kelpjava.j2se.BinaryReader;

/**
 * MNIST のラベルファイルをロードするためのクラス. http://yann.lecun.com/exdb/mnis
 */
public class MnistLabelFile {
	/**
	 * 0x0000 から始まるマジックナンバー. 0x00000801 (2049) が入る.
	 */

	final public int magicNumber;

	/**
	 * ラベルの数.
	 */
	public int numberOfItems;

	/**
	 * ラベルの配列.
	 */
	public byte[] labelList;

	/**
	 * コンストラクタ.
	 */
	public MnistLabelFile(File i_file) throws IOException {
		this(new FileInputStream(i_file));
	}

	/**
	 * MNIST のラベルファイルをロードする. 失敗した時は null を返す.
	 * 
	 * @param i_stream
	 *            ラベルファイルのパス.
	 * @throws IOException
	 */
	public MnistLabelFile(InputStream i_stream) throws IOException {
		BinaryReader br = new BinaryReader(i_stream, BinaryReader.ENDIAN_BIG);
		// バイト配列を分解する
		this.magicNumber = br.getInt();
		this.numberOfItems = br.getInt();
		this.labelList = br.getByteArray(this.numberOfItems);
	}
}
