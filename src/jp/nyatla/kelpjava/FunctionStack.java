package jp.nyatla.kelpjava;

import java.util.ArrayList;
import java.util.List;

import jp.nyatla.kelpjava.common.NdArray;

/**
 * 層を積み上げるこのライブラリのメインとなるクラス。 一回のForward、Backward、Updateで同時に実行される関数の集まり
 * [Serializable]
 */
public class FunctionStack extends Function {
	/**
	 * すべての層がココにFunctionクラスとして保管される
	 */
	final public Function[] functions;
	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	public FunctionStack(FunctionStack i_src) {
		super(i_src);
		this.functions=new Function[i_src.functions.length];
		for (int i=0;i<this.functions.length;i++) {
			this.functions[i]=((Function) i_src.functions[i].deepCopy());
		}
	}


	/**
	 * コンストラクタ
	 * 
	 * @param i_functions
	 */
	public FunctionStack(Function... i_functions) {
		super("FunctionStack");
		// 入力された関数を振り分ける
		this.functions=i_functions;
		
		List<OptimizeParameter> l=new ArrayList<OptimizeParameter>();
		for (int i=0;i<this.parameters.length;i++) {
			for(int j=0;j<this.functions[i].parameters.length;j++){
				l.add(i_functions[i].parameters[j]);
			}
		}
		this.parameters=l.toArray(new OptimizeParameter[0]);
	}


	/**
	 * Functionとして呼び出された時にバトンを渡す
	 * 
	 * @param x
	 * @return
	 */
	@Override
	protected NdArray[] forwardSingle(NdArray[] i_x) {
		return this.forward(i_x);
	}

	/**
	 * Functionとして呼び出された時にバトンを渡す
	 * 
	 * @param gy
	 * @return
	 */
	@Override
	protected NdArray[] backwardSingle(NdArray[] i_gy) {
		return this.backward(i_gy);
	}

	/**
	 * Forward
	 */
	@Override
	public NdArray[] forward(NdArray[] i_input) {
		for (int i = 0; i < this.functions.length; i++) {
			i_input = this.functions[i].forward(i_input);
		}

		return i_input;
	}

	/**
	 * Backward
	 * 
	 * @param backwardResult
	 * @return
	 */
	@Override
	public NdArray[] backward(NdArray[] i_backwardResult) {
		for (int i = this.functions.length - 1; i >= 0; i--) {
			// ここちょっとキモイ
			i_backwardResult = this.functions[i].backward(i_backwardResult);
		}

		return i_backwardResult;
	}

	/**
	 * Forward
	 * 
	 * @param input
	 * @return
	 */
	@Override
	public NdArray forward(NdArray i_input) {
		for (int i = 0; i < this.functions.length; i++) {
			i_input = this.functions[i].forward(i_input);
		}

		return i_input;
	}

	/**
	 * Backward
	 */
	@Override
	public NdArray backward(NdArray backwardResult) {
		for (int i = this.functions.length - 1; i >= 0; i--) {
			backwardResult = this.functions[i].backward(backwardResult);
		}

		return backwardResult;
	}



	/**
	 * 重みの更新処理
	 */
	public void Update(Optimizer[] i_optimizers)
	{
		// 更新実行前に訓練カウントを使って各Functionの傾きを補正
		for (int i = 0; i < this.functions.length; i++) {
			for (int j = 0; j < this.functions[i].parameters.length; j++) {
				for (int k = 0; k < this.functions[i].parameters[j].length(); k++) {
					this.functions[i].parameters[j].grad.data[k] /= this.functions[i].parameters[j].trainCount;
				}
			}
		}

		// Optimizerの更新を実行
		for (Optimizer optimizer : i_optimizers) {
			optimizer.update();
		}

		// 傾きとカウンタをリセット
		this.ClearGrads();

		// ガベージコレクタを明示的に起動
		Runtime.getRuntime().gc();
	}

	/**
	 * 傾きの初期化
	 */
	public void ClearGrads() {
		for (int i = 0; i < this.functions.length; i++) {
			for (int j = 0; j < this.functions[i].parameters.length; j++) {
				this.functions[i].parameters[j].clearGrad();
			}
		}
	}

	/**
	 * ある処理実行後に特定のデータを初期値に戻す処理
	 */
	@Override
	public void resetState() {
		for (int i = 0; i < this.functions.length; i++) {
			this.functions[i].resetState();
		}
	}

	/**
	 * 予想を実行する
	 * 
	 * @param forwardResult
	 * @return
	 */
	@Override
	public NdArray[] predict(NdArray[] forwardResult) {
		for (int i = 0; i < this.functions.length; i++) {
			forwardResult = this.functions[i].predict(forwardResult);
		}

		return forwardResult;
	}

	/**
	 * 予想を実行する[非バッチ]
	 */
	@Override
	public NdArray predict(NdArray input) {
		for (int i = 0; i < this.functions.length; i++) {
			input = this.functions[i].predict(input);
		}

		return input;
	}

	@Override
	public Object deepCopy() {
		return new FunctionStack(this);
	}

	// public void Save(string fileName)
	// {
	// BinaryFormatter bf = new BinaryFormatter();
	//
	// using (Stream stream = File.OpenWrite(fileName))
	// {
	// bf.Serialize(stream, this);
	// }
	// }
	//
	// public static FunctionStack Load(string fileName)
	// {
	// BinaryFormatter bf = new BinaryFormatter();
	// FunctionStack result;
	//
	// using (Stream stream = File.OpenRead(fileName))
	// {
	// result = (FunctionStack)bf.Deserialize(stream);
	// }
	//
	// return result;
	// }
}
