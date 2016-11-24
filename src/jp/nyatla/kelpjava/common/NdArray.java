package jp.nyatla.kelpjava.common;

import java.util.ArrayList;
import java.util.List;

/**
 * NumpyのNdArrayを模したクラス N次元のArrayクラスを入力に取り、内部的には1次元配列として保持する事で動作を模倣している
 * 
 */
public class NdArray implements Cloneable {
	final public double[] data;
	final public int[] shape;

	public NdArray(double[] i_data, int[] i_shape) {
		// コンストラクタはコピーを作成する
		this.data = i_data.clone();
		this.shape = i_shape.clone();
	}

	public NdArray(NdArray i_ndArray) {
		// コンストラクタはコピーを作成する
		this(i_ndArray.data, i_ndArray.shape);
	}

	public int length() {
		return this.data.length;
	}

	public int rank() {
		return this.shape.length;
	}

	// //繰り返し呼び出されるシーンでは使用しないこと
	// public double Get(params int[] indices)
	// {
	// return this.Data[this.GetIndex(indices)];
	// }
	//
	// //繰り返し呼び出されるシーンでは使用しないこと
	// public void Set(int[] indices, double val)
	// {
	// this.Data[this.GetIndex(indices)] = val;
	// }

	public static NdArray zerosLike(NdArray baseArray) {
		return new NdArray(new double[baseArray.length()], baseArray.shape);
	}

	public static NdArray onesLike(NdArray baseArray) {
		double[] resutlArray = new double[baseArray.length()];

		for (int i = 0; i < resutlArray.length; i++) {
			resutlArray[i] = 1;
		}
		return new NdArray(resutlArray, baseArray.shape);
	}

	// public static NdArray Zeros(params int[] shape)
	// {
	// return new NdArray(new double[ShapeToArrayLength(shape)], shape);
	// }
	//
	// public static NdArray Ones(params int[] shape)
	// {
	// double[] resutlArray = new double[ShapeToArrayLength(shape)];
	//
	// for (int i = 0; i < resutlArray.Length; i++)
	// {
	// resutlArray[i] = 1;
	// }
	//
	// return new NdArray(resutlArray, shape);
	// }
	//
	// static int ShapeToArrayLength(params int[] shapes)
	// {
	// int result = 1;
	//
	// foreach (int shape in shapes)
	// {
	// result *= shape;
	// }
	//
	// return result;
	// }
	//
	// public static NdArray[] FromArray(Array[] data)
	// {
	// NdArray[] result = new NdArray[data.Length];
	//
	// for (int i = 0; i < result.Length; i++)
	// {
	// result[i] = FromArray(data[i]);
	// }
	//
	// return result;
	// }
	//
	// public static NdArray FromArray(Array data)
	// {
	// double[] resultData = new double[data.Length];
	// int[] resultShape;
	//
	// if (data.Rank == 1)
	// {
	// //型変換を兼ねる
	// Array.Copy(data, resultData, data.Length);
	//
	// resultShape = new[] { data.Length };
	// }
	// else
	// {
	// //int -> doubleの指定ミスで例外がポコポコ出るので、ここで吸収
	// if (data.GetType().GetElementType() != typeof(double))
	// {
	// Type arrayType = data.GetType().GetElementType();
	// //一次元の長さの配列を用意
	// var array = Array.CreateInstance(arrayType, data.Length);
	// //一次元化して
	// Buffer.BlockCopy(data, 0, array, 0, Marshal.SizeOf(arrayType) *
	// resultData.Length);
	//
	// //型変換しつつコピー
	// Array.Copy(array, resultData, array.Length);
	// }
	// else
	// {
	// Buffer.BlockCopy(data, 0, resultData, 0, sizeof(double) *
	// resultData.Length);
	// }
	//
	// resultShape = new int[data.Rank];
	// for (int i = 0; i < data.Rank; i++)
	// {
	// resultShape[i] = data.GetLength(i);
	// }
	// }
	//
	// return new NdArray(resultData, resultShape);
	// }

	public void fill(double val) {
		for (int i = 0; i < this.data.length; i++) {
			this.data[i] = val;
		}
	}

	// //N次元のIndexから１次元のIndexを取得する
	// private int GetIndex(params int[] indices)
	// {
	// #if DEBUG
	// if (this.Shape.Length != indices.Length)
	// {
	// throw new Exception("次元数がマッチしていません");
	// }
	// #endif
	//
	// int index = 0;
	//
	// for (int i = 0; i < indices.Length; i++)
	// {
	// #if DEBUG
	// if (this.Shape[i] <= indices[i])
	// {
	// throw new Exception(i + "次元の添字が範囲を超えています");
	// }
	// #endif
	//
	// int rankOffset = 1;
	//
	// for (int j = i + 1; j < this.Shape.Length; j++)
	// {
	// rankOffset *= this.Shape[j];
	// }
	//
	// index += indices[i] * rankOffset;
	// }
	//
	// return index;
	// }

	/**
	 * Numpyっぽく値を文字列に変換して出力する
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		int intMaxLength = 0; // 整数部の最大値
		int realMaxLength = 0; // 小数点以下の最大値
		boolean isExponential = false; // 指数表現にするか

		for (int i = 0; i < this.data.length; i++) {
			String[] divStr = Double.toString(this.data[i]).split(".");
			intMaxLength = Math.max(intMaxLength, divStr[0].length());
			if (divStr.length > 1 && !isExponential) {
				isExponential = divStr[1].contains("E");
			}

			if (realMaxLength != 8 && divStr.length == 2) {
				realMaxLength = Math.max(realMaxLength, divStr[1].length());
				if (realMaxLength > 8)
					realMaxLength = 8;
			}
		}

		// 配列の約数を取得
		List<Integer> CommonDivisorList = new ArrayList<Integer>();

		// 一個目は手動取得
		CommonDivisorList.add(this.shape[this.shape.length - 1]);
		for (int i = 1; i < this.shape.length; i++) {
			CommonDivisorList.add(CommonDivisorList.get(CommonDivisorList
					.size() - 1) * this.shape[this.shape.length - i - 1]);
		}

		// 先頭の括弧
		for (int i = 0; i < this.shape.length; i++) {
			sb.append("[");
		}

		int closer = 0;
		for (int i = 0; i < this.length(); i++) {
			String[] divStr;
			if (isExponential) {
				// 代替手法がよくわからない。
				// divStr = this.Data[i].ToString("0.00000000e+00").Split('.');
				divStr = Double.toString(this.data[i]).split(".");
			} else {
				divStr = Double.toString(this.data[i]).split(".");
			}

			// 最大文字数でインデントを揃える
			for (int j = 0; j < intMaxLength - divStr[0].length(); j++) {
				sb.append(" ");
			}
			sb.append(divStr[0]);
			if (realMaxLength != 0) {
				sb.append(".");
			}
			if (divStr.length == 2) {
				sb.append(divStr[1].length() > 8 && !isExponential ? divStr[1]
						.substring(0, 8) : divStr[1]);
				for (int j = 0; j < realMaxLength - divStr[1].length(); j++) {
					sb.append(" ");
				}
			} else {
				for (int j = 0; j < realMaxLength; j++) {
					sb.append(" ");
				}
			}

			// 約数を調査してピッタリなら括弧を出力
			if (i != this.length() - 1) {
				for (int j = 0; j < CommonDivisorList.size(); j++) {
					int commonDivisor =CommonDivisorList.get(j);
					if ((i + 1) % commonDivisor == 0) {
						sb.append("]");
						closer++;
					}
				}

				sb.append(" ");

				if ((i + 1) % CommonDivisorList.get(0) == 0) {
					// 閉じ括弧分だけ改行を追加
					for (int j = 0; j < closer; j++) {
						sb.append("\n");
					}
					closer = 0;

					// 括弧前のインデント
					for (int j = 0; j < CommonDivisorList.size(); j++) {
						int commonDivisor =CommonDivisorList.get(j);
						if ((i + 1) % commonDivisor != 0) {
							sb.append(" ");
						}
					}
				}

				for (int j = 0; j < CommonDivisorList.size(); j++) {
					int commonDivisor =CommonDivisorList.get(j);
					if ((i + 1) % commonDivisor == 0) {
						sb.append("[");
					}
				}
			}
		}

		// 後端の括弧
		for (int i = 0; i < this.shape.length; i++) {
			sb.append("]");
		}
		return sb.toString();
	}

	@Override
	public Object clone() {
		return new NdArray(this);
	}
}
