package jp.nyatla.kelpjava;

import java.util.ArrayList;
import java.util.List;

import jp.nyatla.kelpjava.common.NdArray;

    //FunctionStackに積み上げるFunctionの基底クラス
    //[Serializable]
    public abstract class Function
    {
        public String Name;

        public List<OptimizeParameter> Parameters = new ArrayList<OptimizeParameter>();

        protected int OutputCount;
        protected int InputCount;
        protected Function(String i_name)
        {
        	this(i_name,0,0);
        }

        //コンストラクタ
        protected Function(String i_name, int i_inputCount, int i_oututCount)
        {
            this.Name = i_name;
            this.InputCount = i_inputCount;
            this.OutputCount = i_oututCount;
        }

        //外部公開用
        public NdArray[] Forward(NdArray[] i_x)
        {
            return this.ForwardSingle(i_x);
        }

        public NdArray[] Backward(NdArray[] i_gy)
        {
            //バッチは内部で割引を行うためgy.Lengthでの加算の必要がない
        	for(int i=0;i<this.Parameters.size();i++){
        		this.Parameters.get(i).trainCount++;
        	}
            return this.BackwardSingle(i_gy);
        }

        //通常であれば非バッチ呼び出しを仮想とするが、
        //バッチ専用関数がスタンダードで非バッチ関数がイレギュラーであるため
        protected abstract NdArray[] ForwardSingle(NdArray[] x);
        protected abstract NdArray[] BackwardSingle(NdArray[] gy);

        //外部公開用非バッチ関数
        public NdArray Forward(NdArray i_x)
        {
            return this.ForwardSingle(i_x);
        }

        public NdArray Backward(NdArray i_gy)
        {
        	for(int i=0;i<this.Parameters.size();i++){
        		this.Parameters.get(i).trainCount++;
        	}
            return this.BackwardSingle(i_gy);
        }

        //任意で個別に非バッチ関数が書けるように用意
        protected NdArray ForwardSingle(NdArray x)
        {
        	NdArray[] na={x};
            return this.ForwardSingle(na)[0];
        }

        protected NdArray BackwardSingle(NdArray gy)
        {
        	NdArray[] na={gy};
            return this.BackwardSingle(na)[0];
        }

        //評価関数
        public NdArray[] Predict(NdArray[] input)
        {
            return this.ForwardSingle(input);
        }

        public NdArray Predict(NdArray input)
        {
            return this.ForwardSingle(input);
        }

        //ある処理実行後に特定のデータを初期値に戻す処理
        public void ResetState()
        {
        	return;
        }

        /**
         * 名前を返します。
         * @return
         */
        @Override
        public String toString()
        {
            return this.Name;
        }
        protected void InitWeight(NdArray array)
        {
        	this.InitWeight(array,1.0);
        }

        //初期値が入力されなかった場合、この関数で初期化を行う
        protected void InitWeight(NdArray array, double masterScale)
        {
            double localScale = 1 / Math.sqrt(2);
            int fanIn = this.GetFans(array.shape);
            double s = localScale * Math.sqrt(2.0 / fanIn);

            for (int i = 0; i < array.length(); i++)
            {
                array.data[i] = this.Normal(s) * masterScale;
            }
        }


        
        private double Normal(double scale)
        {
            Mother.Sigma = scale;
            return Mother.RandomNormal();
        }

        private int GetFans(int[] shape)
        {
            int result = 1;

            for (int i = 1; i < shape.length; i++)
            {
                result *= shape[i];
            }

            return result;
        }
    }
