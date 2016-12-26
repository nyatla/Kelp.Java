package jp.nyatla.kelpjava.io.vocabulary;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class IndexedTextData
{
	public IndexedTextData()
	{
	}
	Map<String,Integer> dic=new HashMap<String,Integer>();
	public void add(String[] i_words)
	{
		//辞書に登録
		for(int i=0;i<i_words.length;i++){
			if(this.dic.containsKey(i_words[i])){
				continue;
			}
			this.dic.put(i_words[i],this.dic.size());
		}
	}
	/**
	 * テキスト配列をId配列に変換します。
	 * @param i_file
	 * @return
	 */
	public int[] getTextIds(String[] i_words)
	{
		List<Integer> dest=new ArrayList<Integer>();
		for(int i=0;i<i_words.length;i++){
			Integer idx=this.dic.get(i_words[i]);
			if(idx==null){
				continue;
			}
			dest.add(idx);
		}
		int[] d=new int[dest.size()];
		for(int i=0;i<d.length;i++){
			d[i]=dest.get(i);
		}
		return d;
	}
	public int[][] getTextIds(String[][] i_words)
	{
		int[][] ret=new int[i_words.length][];
		for(int i=0;i<ret.length;i++){
			ret[i]=this.getTextIds(i_words[i]);
		}
		return ret;
	}

	
	public int getId(String i_string) {
		return this.dic.get(i_string);
	}	
	public void add(String[][] lines)
	{
		for(String[] i:lines)
		{
			this.add(i);
		}
	}
	
	/**
	 * 辞書の長さを返します。
	 * @return
	 */
	public int getLength() {
		return this.dic.size();
	}

	
}
