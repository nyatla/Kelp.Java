package jp.nyatla.kelpjava.test;

import java.io.File;
import java.io.IOException;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.functions.activations.ReLU;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.functions.normalization.BatchNormalization;
import jp.nyatla.kelpjava.io.MnistData;
import jp.nyatla.kelpjava.io.MnistData.DataSet;
import jp.nyatla.kelpjava.loss.SoftmaxCrossEntropy;
import jp.nyatla.kelpjava.optimizers.AdaGrad;


    //バッチノーマライゼーションを使った15層MLPによるMNIST（手書き文字）の学習
    //参考： http://takatakamanbou.hatenablog.com/entry/2015/12/20/233232
    public class Test7
    {
        //ミニバッチの数
        //ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
        final static int BATCH_DATA_COUNT = 128;

        //一世代あたりの訓練回数
        final static int TRAIN_DATA_COUNT = 50000;

        //性能評価時のデータ数
        final static int TEST_DATA_COUNT = 200;

        //中間層の数
        final static int N = 30; //参考先リンクと同様の1000でも動作するがCPUでは遅いので

        public static void main(String[] args) throws IOException
        {
            //MNISTのデータを用意する
        	System.out.println("MNIST Data Loading...");
    		MnistData train = new MnistData(new File("data/train-images.idx3-ubyte"), new File("data/train-labels.idx1-ubyte"));
    		MnistData teach = new MnistData(new File("data/t10k-images.idx3-ubyte"), new File("data/t10k-labels.idx1-ubyte"));

            System.out.println("Training Start...");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(28 * 28, N,"l1 Linear"), // L1
                new BatchNormalization(N,"l1 BatchNorm"),
                new ReLU("l1 ReLU"),
                new Linear(N, N,"l2 Linear"), // L2
                new BatchNormalization(N,"l2 BatchNorm"),
                new ReLU("l2 ReLU"),
                new Linear(N, N,"l3 Linear"), // L3
                new BatchNormalization(N,"l3 BatchNorm"),
                new ReLU("l3 ReLU"),
                new Linear(N, N,"l4 Linear"), // L4
                new BatchNormalization(N,"l4 BatchNorm"),
                new ReLU("l4 ReLU"),
                new Linear(N, N,"l5 Linear"), // L5
                new BatchNormalization(N,"l5 BatchNorm"),
                new ReLU("l5 ReLU"),
                new Linear(N, N,"l6 Linear"), // L6
                new BatchNormalization(N, "l6 BatchNorm"),
                new ReLU("l6 ReLU"),
                new Linear(N, N,"l7 Linear"), // L7
                new BatchNormalization(N,"l7 BatchNorm"),
                new ReLU("l7 ReLU"),
                new Linear(N, N,"l8 Linear"), // L8
                new BatchNormalization(N,"l8 BatchNorm"),
                new ReLU("l8 ReLU"),
                new Linear(N, N,"l9 Linear"), // L9
                new BatchNormalization(N,"l9 BatchNorm"),
                new ReLU("l9 ReLU"),
                new Linear(N, N, "l10 Linear"), // L10
                new BatchNormalization(N,"l10 BatchNorm"),
                new ReLU("l10 ReLU"),
                new Linear(N, N,"l11 Linear"), // L11
                new BatchNormalization(N,"l11 BatchNorm"),
                new ReLU("l11 ReLU"),
                new Linear(N, N,"l12 Linear"), // L12
                new BatchNormalization(N,"l12 BatchNorm"),
                new ReLU("l12 ReLU"),
                new Linear(N, N,"l13 Linear"), // L13
                new BatchNormalization(N,"l13 BatchNorm"),
                new ReLU("l13 ReLU"),
                new Linear(N, N,"l14 Linear"), // L14
                new BatchNormalization(N,"l14 BatchNorm"),
                new ReLU("l14 ReLU"),
                new Linear(N, 10,"l15 Linear") // L15
            );


            Trainer trainer=new Trainer();
           //optimizerを宣言
            nn.setOptimizer(new AdaGrad());

            //三世代学習
            for (int epoch = 0; epoch < 3; epoch++)
            {
                System.out.println("epoch " + (epoch + 1));

                //全体での誤差を集計
                double[] totalLoss = new double[TRAIN_DATA_COUNT];

                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //訓練データからランダムにデータを取得
                    DataSet datasetX = train.getRandomDataSet(BATCH_DATA_COUNT);

                    //学習を実行
                    double sumLoss = trainer.batchTrain(nn, datasetX.image, datasetX.label, new SoftmaxCrossEntropy());
                    totalLoss[i-1]=sumLoss;

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        //結果出力
                        System.out.println("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                        System.out.println("total loss " + JavaUtils.average(totalLoss,i));
                        System.out.println("local loss " + sumLoss);
                        System.out.println("");
                        System.out.println("Testing...");

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

