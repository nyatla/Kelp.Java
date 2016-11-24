package jp.nyatla.kelpjava;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import jp.nyatla.kelpjava.common.IDuplicatable;

/**
 * Optimizerの基底クラス [Serializable]
 */
public abstract class Optimizer implements IDuplicatable
{
	/**
	 * コピーコンストラクタ
	 * 
	 * @param i_src
	 */
	protected Optimizer(Optimizer i_src)
	{
		this.updateCount = i_src.updateCount;
		for(OptimizeParameter i:i_src.parameters){
			this.parameters.add((OptimizeParameter) i.deepCopy());
		}
	}

	/**
	 * 更新回数のカウント
	 */
	protected long updateCount = 1;
	final protected List<OptimizeParameter> parameters = new ArrayList<OptimizeParameter>();

	/**
	 * 更新回数のカウントを取りつつ更新処理を呼び出す
	 */
	public void update() {
		this.doUpdate();
		this.updateCount++;
	}

	/**
	 * カウントを取るために呼び変えしている
	 */
	protected abstract void doUpdate();

	/**
	 * 更新対象となるパラメータを保存
	 * @param parameters
	 */
	public void SetParameters(Collection<OptimizeParameter> parameters) {
		this.parameters.addAll(parameters);
		this.initialize();
	}

	protected void initialize() {
	}
	
	public static Optimizer[] deepCopy(Optimizer[] i_src)
	{
		Optimizer[] r=new Optimizer[i_src.length];
		for(int i=0;i<r.length;i++){
			r[i]=(Optimizer) i_src[i].deepCopy();
		}
		return r;
	}	
	
}
