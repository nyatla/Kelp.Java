package jp.nyatla.kelpjava.test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.activations.ReLU;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.functions.normalization.BatchNormalization;
import jp.nyatla.kelpjava.io.mnist.MnistData;
import jp.nyatla.kelpjava.io.mnist.MnistData.DataSet;
import jp.nyatla.kelpjava.loss.LossFunction;
import jp.nyatla.kelpjava.loss.MeanSquaredError;
import jp.nyatla.kelpjava.loss.SoftmaxCrossEntropy;
import jp.nyatla.kelpjava.optimizers.Adam;

    /**
     * Decoupled Neural Interfaces using Synthetic GradientsによるMNIST（手書き文字）の学習
     * 教師信号にラベル情報を混ぜ込むcDNIモデルDと表現されている全層のDecoupledを非同期で実行
     * http://ralo23.hatenablog.com/entry/2016/10/22/233405
     */
    class Test12
    {
        //ミニバッチの数
        //ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
    	final static int BATCH_DATA_COUNT = 256;

        //一世代あたりの訓練回数
    	final static int TRAIN_DATA_COUNT = 234; // = 60,000 / 256

        //性能評価時のデータ数
    	final static int TEST_DATA_COUNT = 100;

        static class ResultDataSet
        {
            public NdArray[] Result;
            public NdArray[] Label;

            public ResultDataSet(NdArray[] result, NdArray[] label)
            {
                this.Result = result;
                this.Label = label;
            }

            public NdArray[] GetTrainData()
            {
                //第一層の傾きを取得
                double[][] train = new double[BATCH_DATA_COUNT][];

                for (int k = 0; k < BATCH_DATA_COUNT; k++)
                {
                    train[k] = new double[256 + 10];
                    train[k][256 + (int)this.Label[k].data[0]] = 1.0;
                    System.arraycopy(this.Result[k].data, 0, train[k], 0,256);
                }

                return NdArray.fromArray(train);
            }
        }

        public static void main(String[] args) throws IOException
        {
            //MNISTのデータを用意する
            System.out.println("MNIST Data Loading...");
    		MnistData train = new MnistData(new File("data/mnist/train-images.idx3-ubyte"), new File("data/mnist/train-labels.idx1-ubyte"));
    		MnistData teach = new MnistData(new File("data/mnist/t10k-images.idx3-ubyte"), new File("data/mnist/t10k-labels.idx1-ubyte"));


            System.out.println("Training Start...");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack Layer1 = new FunctionStack(
                new Linear(28 * 28, 256, "l1 Linear"),
                new BatchNormalization(256, "l1 Norm"),
                new ReLU("l1 ReLU")
            );

            FunctionStack Layer2 = new FunctionStack(
                new Linear(256, 256, "l2 Linear"),
                new BatchNormalization(256, "l2 Norm"),
                new ReLU("l2 ReLU")
            );

            FunctionStack Layer3 = new FunctionStack(
                new Linear(256, 256, "l3 Linear"),
                new BatchNormalization(256, "l3 Norm"),
                new ReLU("l3 ReLU")
            );

            FunctionStack Layer4 = new FunctionStack(
                new Linear(256, 10, "l4 Linear")
            );

            //FunctionStack自身もFunctionとして積み上げられる
            FunctionStack nn = new FunctionStack
            (
                Layer1,
                Layer2,
                Layer3,
                Layer4
            );

            FunctionStack cDNI1 = new FunctionStack(
                new Linear(256 + 10, 1024, "cDNI1 Linear1"),
                new BatchNormalization(1024, "cDNI1 Nrom1"),
                new ReLU("cDNI1 ReLU1"),
                new Linear(1024, 256, false,new NdArray(new double[1024][256]),null, "DNI1 Linear3")
            );

            FunctionStack cDNI2 = new FunctionStack(
                new Linear(256+10, 1024, "cDNI2 Linear1"),
                new BatchNormalization(1024, "cDNI2 Nrom1"),
                new ReLU("cDNI2 ReLU1"),
                new Linear(1024, 256, false,new NdArray(new double[1024][256]),null,"cDNI2 Linear3")
            );

            FunctionStack cDNI3 = new FunctionStack(
                new Linear(256+10, 1024, "cDNI3 Linear1"),
                new BatchNormalization(1024, "cDNI3 Nrom1"),
                new ReLU("cDNI3 ReLU1"),
                new Linear(1024, 256, false,new NdArray(new double[1024][256]),null, "cDNI3 Linear3")
            );

            //optimizerを宣言
            Layer1.setOptimizer(new Adam(0.00003));
            Layer2.setOptimizer(new Adam(0.00003));
            Layer3.setOptimizer(new Adam(0.00003));
            Layer4.setOptimizer(new Adam(0.00003));

            cDNI1.setOptimizer(new Adam(0.00003));
            cDNI2.setOptimizer(new Adam(0.00003));
            cDNI3.setOptimizer(new Adam(0.00003));
            Trainer trainer=new Trainer();
            for (int epoch = 0; epoch < 10; epoch++)
            {
                System.out.println("epoch " + (epoch + 1));

                //全体での誤差を集計
                List<Double> totalLoss = new ArrayList<Double>();

                List<Double> cDNI1totalLoss = new ArrayList<Double>();

                List<Double> cDNI2totalLoss = new ArrayList<Double>();

                List<Double> cDNI3totalLoss = new ArrayList<Double>();


                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //訓練データからランダムにデータを取得
                    DataSet datasetX = train.getRandomDataSet(BATCH_DATA_COUNT);

                    //第一層を実行
                    NdArray[] layer1ForwardResult = Layer1.forward(datasetX.image);
                    ResultDataSet layer1ResultDataSet = new ResultDataSet(layer1ForwardResult, datasetX.label);

                    ////第一層の傾きを取得
                    NdArray[] cDNI1Result = cDNI1.forward(layer1ResultDataSet.GetTrainData());

                    //第一層を更新
                    Layer1.backward(cDNI1Result);
                    Layer1.update();


                    //第二層を実行
                    NdArray[] layer2ForwardResult = Layer2.forward(layer1ResultDataSet.Result);
                    ResultDataSet layer2ResultDataSet = new ResultDataSet(layer2ForwardResult, layer1ResultDataSet.Label);

                    //第二層の傾きを取得
                    NdArray[] cDNI2Result =cDNI2.forward(layer2ResultDataSet.GetTrainData());

                    //第二層を更新
                    NdArray[] layer2BackwardResult = Layer2.backward(cDNI2Result);
                    Layer2.update();


                    //第一層用のcDNIの学習を実行
//                    double cDNI1loss = 0;
                    LossFunction.Results DNI1lossResult = new MeanSquaredError().evaluate(cDNI1Result, layer2BackwardResult);

                    cDNI1.backward(DNI1lossResult.data);
                    cDNI1.update();
                    cDNI1totalLoss.add(DNI1lossResult.loss);


                    //第三層を実行
                    NdArray[] layer3ForwardResult = Layer3.forward(layer2ResultDataSet.Result);
                    ResultDataSet layer3ResultDataSet = new ResultDataSet(layer3ForwardResult, layer2ResultDataSet.Label);

                    //第三層の傾きを取得
                    NdArray[] cDNI3Result = cDNI3.forward(layer3ResultDataSet.GetTrainData());

                    //第三層を更新
                    NdArray[] layer3BackwardResult = Layer3.backward(cDNI3Result);
                    Layer3.update();


                    //第二層用のcDNIの学習を実行
                    LossFunction.Results DNI2lossResult = new MeanSquaredError().evaluate(cDNI2Result, layer3BackwardResult);

                    cDNI2.backward(DNI2lossResult.data);
                    cDNI2.update();
                    cDNI2totalLoss.add(DNI2lossResult.loss);


                    //第四層を実行
                    NdArray[] layer4ForwardResult = Layer4.forward(layer3ResultDataSet.Result);

                    //第四層の傾きを取得
                    LossFunction.Results lossResult = new SoftmaxCrossEntropy().evaluate(layer4ForwardResult, layer3ResultDataSet.Label);

                    //第四層を更新
                    NdArray[] layer4BackwardResult = Layer4.backward(lossResult.data);
                    Layer4.update();
                    totalLoss.add(lossResult.loss);


                    //第三層用のcDNIの学習を実行
                    LossFunction.Results DNI3lossResult = new MeanSquaredError().evaluate(cDNI3Result, layer4BackwardResult);

                    cDNI3.backward(DNI3lossResult.data);
                    cDNI3.update();
                    cDNI3totalLoss.add(DNI3lossResult.loss);


                    System.out.println("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                    //結果出力
                    System.out.println("total loss " + JavaUtils.average(totalLoss));
                    System.out.println("local loss " + lossResult.loss);

                    System.out.println("\ncDNI1 total loss " + JavaUtils.average(cDNI1totalLoss));
                    System.out.println("cDNI2 total loss " + JavaUtils.average(cDNI2totalLoss));
                    System.out.println("cDNI3 total loss " + JavaUtils.average(cDNI3totalLoss));

                    System.out.println("\ncDNI1 local loss " + DNI1lossResult.loss);
                    System.out.println("cDNI2 local loss " + DNI2lossResult.loss);
                    System.out.println("cDNI3 local loss " + DNI3lossResult.loss);

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                    	System.out.println("\nTesting...");

                        //テストデータからランダムにデータを取得
                        DataSet datasetY = teach.getRandomDataSet(TEST_DATA_COUNT);

                        //テストを実行
                        double accuracy = trainer.accuracy(nn, datasetY.image, datasetY.label);
                        System.out.println("accuracy " + accuracy);
                    }
                }
            }
        }
    }

