package jp.nyatla.kelpjava.io.vocabulary;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

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

	public VocabularyText(InputStream i_stream) throws IOException
	{
		List<String> dest=new ArrayList<String>();
		byte[] bin = BinaryReader.toArray(i_stream);
		String str = new String(bin, "UTF-8");
		BufferedReader br=new BufferedReader(new StringReader(str));
		for(String l=br.readLine();l!=null;l=br.readLine()){
			String[] ls=l.split(" ");
			for(int i=0;i<ls.length;i++){
				if(ls[i].trim().isEmpty()){
					continue;
				}
				dest.add(ls[i]);
			}
			dest.add(EOS);
		}		
		this.text = dest.toArray(new String[0]);
		return;
	}


	public static void main(String[] args) throws IOException {
		VocabularyText v = new VocabularyText(
				new File("data/ptb/ptb.train.txt"));
		return;

	}
}
