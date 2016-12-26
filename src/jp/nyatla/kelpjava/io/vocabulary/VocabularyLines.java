package jp.nyatla.kelpjava.io.vocabulary;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * テキストデータの
 * 
 */
public class VocabularyLines {
	final public String[][] lines;
	public VocabularyLines(File i_file) throws IOException {
		this(new VocabularyText(new FileInputStream(i_file)));
	}

	public VocabularyLines(VocabularyText i_src) throws IOException
	{
		String[] s =i_src.text;
		//Lineに変換
		List<String[]> l=new ArrayList<String[]>();
		List<String> w=new ArrayList<String>();
		for(int i=0;i<s.length;i++){
			w.add(s[i]);
			if(s[i].compareTo(VocabularyText.EOS)==0){
				l.add(w.toArray(new String[0]));
				w.clear();
			}
		}
		this.lines=l.toArray(new String[0][0]);
	}


	public static void main(String[] args) throws IOException {
		VocabularyLines v = new VocabularyLines(
				new File("data/ptb/ptb.train.txt"));
		return;

	}
}
