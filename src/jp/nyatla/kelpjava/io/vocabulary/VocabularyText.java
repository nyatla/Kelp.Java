package jp.nyatla.kelpjava.io.vocabulary;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import jp.nyatla.kelpjava.j2se.BinaryReader;

/**
 * テキストデータの
 * 
 */
public class VocabularyText {
	final public static String EOS="<EOS>";
	final public String[] text;
	public VocabularyText(File i_file) throws IOException {
		this(new FileInputStream(i_file));
	}

	public VocabularyText(InputStream i_stream) throws IOException {
		byte[] bin = BinaryReader.toArray(i_stream);
		String str = new String(bin, "UTF-8");
		this.text = str.replace("\r\n", EOS).trim().split(" ");
		return;
	}


	public static void main(String[] args) throws IOException {
		VocabularyText v = new VocabularyText(
				new File("data/ptb/ptb.train.txt"));
		return;

	}
}
