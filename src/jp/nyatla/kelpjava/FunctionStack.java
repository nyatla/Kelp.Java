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
	final public List<Function> functions = new ArrayList<Function>();
	/**
	 * Optimizerをココで保持する
	 */
	private Optimizer[] optimizers;

	/**
	 * コンストラクタ
	 * 
	 * @param i_functions
	 */
	public FunctionStack(Function... i_functions) {
		super("FunctionStack");
		// 入力された関数を振り分ける
		for (Function function : i_functions) {
			// 全関数リストへ追加
			this.functions.add(function);

			// パラメーターを保持
			this.parameters.addAll(function.parameters);
		}
	}

	public FunctionStack(FunctionStack i_src) {
		super(i_src);
		for (Function i : i_src.functions) {
			this.functions.add((Function) i.deepCopy());
		}
		if (i_src.optimizers != null) {
			this.optimizers = Optimizer.deepCopy(i_src.optimizers);
		}
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
		for (int i = 0; i < this.functions.size(); i++) {
			i_input = this.functions.get(i).forward(i_input);
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
		for (int i = this.functions.size() - 1; i >= 0; i--) {
			// ここちょっとキモイ
			i_backwardResult = this.functions.get(i).backward(i_backwardResult);
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
		for (int i = 0; i < this.functions.size(); i++) {
			i_input = this.functions.get(i).forward(i_input);
		}

		return i_input;
	}

	/**
	 * Backward
	 */
	@Override
	public NdArray backward(NdArray backwardResult) {
		for (int i = this.functions.size() - 1; i >= 0; i--) {
			backwardResult = this.functions.get(i).backward(backwardResult);
		}

		return backwardResult;
	}

	// Optimizerを設定
	public void SetOptimizer(Optimizer... i_optimizers) {
		this.optimizers = i_optimizers;
		for (Optimizer optimizer : i_optimizers) {
			optimizer.SetParameters(this.parameters);
		}
	}

	/**
	 * 重みの更新処理
	 */
	public void Update() {
		// 更新実行前に訓練カウントを使って各Functionの傾きを補正
		for (int i = 0; i < this.functions.size(); i++) {
			for (int j = 0; j < this.functions.get(i).parameters.size(); j++) {
				for (int k = 0; k < this.functions.get(i).parameters.get(j)
						.length(); k++) {
					this.functions.get(i).parameters.get(j).grad.data[k] /= this.functions
							.get(i).parameters.get(j).trainCount;
				}
			}
		}

		// Optimizerの更新を実行
		for (Optimizer optimizer : this.optimizers) {
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
		for (int i = 0; i < this.functions.size(); i++) {
			for (int j = 0; j < this.functions.get(i).parameters.size(); j++) {
				this.functions.get(i).parameters.get(j).clearGrad();
			}
		}
	}

	/**
	 * ある処理実行後に特定のデータを初期値に戻す処理
	 */
	@Override
	public void resetState() {
		for (int i = 0; i < this.functions.size(); i++) {
			this.functions.get(i).resetState();
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
		for (int i = 0; i < this.functions.size(); i++) {
			forwardResult = this.functions.get(i).predict(forwardResult);
		}

		return forwardResult;
	}

	/**
	 * 予想を実行する[非バッチ]
	 */
	@Override
	public NdArray predict(NdArray input) {
		for (int i = 0; i < this.functions.size(); i++) {
			input = this.functions.get(i).predict(input);
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
