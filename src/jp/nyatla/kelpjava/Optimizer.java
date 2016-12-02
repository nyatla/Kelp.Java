package jp.nyatla.kelpjava;

/**
 * Optimizerの基底クラス [Serializable]
 */
public abstract class Optimizer
{	
	/**
	 * 更新回数のカウント
	 */
	protected long updateCount = 1;
	final protected OptimizeParameter[] parameters;

    protected Optimizer(OptimizeParameter[] i_parameters)
    {
        this.parameters = i_parameters;
    }	



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
	
//	public static Optimizer[] deepCopy(Optimizer[] i_src)
//	{
//		Optimizer[] r=new Optimizer[i_src.length];
//		for(int i=0;i<r.length;i++){
//			r[i]=(Optimizer) i_src[i].deepCopy();
//		}
//		return r;
//	}	
	
}
