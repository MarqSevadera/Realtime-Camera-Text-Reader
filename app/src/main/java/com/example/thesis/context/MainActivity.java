package com.example.thesis.context;

import android.app.Activity;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements JavaCameraView.CvCameraViewListener2 {

    BaseLoaderCallback baseLoaderCallback;
    JavaCameraView cameraBridgeViewBase;
    Mat mat1,mat2,mat3;

    private int upperH, upperS, upperV, lowerH, lowerS, lowerV;
    Toast toast;
    private Scalar scalarLow, scalarHigh;
    private ImageButton btnBlue, btnRed;

    private boolean mReadyForRecognition = true;
    private Bitmap mBitmapToRecognize;
    private TextToSpeech mTextToSpeech;
    int ttsInitResult;

    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        btnBlue = findViewById(R.id.btnBlue);
        btnRed = findViewById(R.id.btnRed);

        upperH = 135;
        upperS = 255;
        upperV = 255;
        lowerH = 105;
        lowerS = 150;
        lowerV = 40;

        cameraBridgeViewBase = findViewById(R.id.myCameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCameraIndex(0);

        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.enableView();


        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);
                switch (status) {
                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }

            }
        };

        mTextToSpeech = new TextToSpeech(MainActivity.this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {

                if(status == TextToSpeech.SUCCESS){
                    ttsInitResult = mTextToSpeech.setLanguage(Locale.US);
                    mTextToSpeech.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                        @Override
                        public void onStart(String utteranceId) { }

                        @Override
                        public void onDone(String utteranceId) {
                            mReadyForRecognition = true;
                        }

                        @Override
                        public void onError(String utteranceId) { }
                    });
                }else{
                    showToast("Feature not supported!");
                }

            }
        });

    }

    public void onCameraViewStopped() {
        mat1.release();
        mat2.release();
    }

    public void onCameraViewStarted(int width, int height) {
        mat1 = new Mat(width, height, CvType.CV_16UC4);
        mat2 = new Mat(width, height, CvType.CV_16UC4);
        mat3 = new Mat(width, height, CvType.CV_16UC4);

    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        btnBlue.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                upperH = 135;
                upperS = 255;
                upperV = 255;
                lowerH = 105;
                lowerS = 150;
                lowerV = 40;
            }
        });

        btnRed.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                upperH = 20;
                upperS = 255;
                upperV = 255;
                lowerH = 0;
                lowerS = 48;
                lowerV = 80;
            }
        });

        scalarLow = new Scalar(lowerH, lowerS,lowerV);
        scalarHigh = new Scalar(upperH,upperS,upperV);

        // check if ready for recognition
        Mat frameMat = inputFrame.rgba();
        if(mReadyForRecognition){
            mReadyForRecognition = false;
            mBitmapToRecognize = Bitmap.createBitmap(frameMat.width() , frameMat.height() , Bitmap.Config.ARGB_8888);
            Point  centerPoint = detectPointer(inputFrame);
            Utils.matToBitmap(inputFrame.rgba(), mBitmapToRecognize);
            runTextRecognition(mBitmapToRecognize, centerPoint);
            return mat1;
        }

        return inputFrame.rgba();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase!=null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
           showToast("OpenCv Error");
        } else {
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
            cameraBridgeViewBase.enableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null) {
            cameraBridgeViewBase.disableView();
        }
    }

    private void showToast(String message){
        Toast.makeText(this , message, Toast.LENGTH_SHORT).show();
    }


    private Point detectPointer(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        Point rectTopCenter = null;

        Imgproc.cvtColor(inputFrame.rgba(), mat1, Imgproc.COLOR_RGB2HSV);
        Core.inRange(mat1, scalarLow, scalarHigh, mat2);
        Size kSize = new Size(7, 7);
        Imgproc.GaussianBlur(mat2, mat2, kSize, 2, 2);
        double otsu;
        otsu = Imgproc.threshold(mat2, mat2, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        Imgproc.Canny(mat2, mat2, otsu, otsu * 2, 3, true);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mat2, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for ( int contourIdx=0; contourIdx < contours.size(); contourIdx++ )

        {
            // Minimum size allowed for consideration
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(contourIdx).toArray() );
            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint( approxCurve.toArray() );

            // Get bounding rect of contour
            Rect rect = Imgproc.boundingRect(points);

            rectTopCenter = new Point( (rect.x + rect.width/2), (rect.y) );

            if (rect.width < 350 && rect.width > 10 && rect.height > 75) {
                Scalar red = new Scalar(5, 255, 50);
                Imgproc.rectangle(mat1, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(90,255,255), 2);
                Imgproc.circle(mat1, rectTopCenter, 3, red, 4);
            }
        }

        Imgproc.cvtColor(mat1, mat1, Imgproc.COLOR_HSV2RGB);
        return rectTopCenter;
    }


    private void runTextRecognition(Bitmap bitmap, final Point point){

        FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(bitmap);
        FirebaseVisionTextRecognizer recognizer = FirebaseVision.getInstance()
                .getOnDeviceTextRecognizer();

        recognizer.processImage(image)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseVisionText>() {
                            @Override
                            public void onSuccess(FirebaseVisionText texts) {

                                processTextRecognitionResult(texts , point);
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                // Task failed with an exception
                                e.printStackTrace();
                            }
                        });
    }


    private android.graphics.Point getMidPoint(android.graphics.Point a , android.graphics.Point b){
        int x = (int)(a.x + b.x)/2;
        int y =  (int) (a.y + b.y)/2;
        return new android.graphics.Point(x,y);
    }

    private double getDistance(android.graphics.Point  a , android.graphics.Point b){
        double x = Math.pow((b.x - a.x) , 2);
        double y = Math.pow((b.y - a.y) , 2);
        return Math.sqrt(x+y);
    }

    private void processTextRecognitionResult(FirebaseVisionText texts, Point point) {

        List<FirebaseVisionText.TextBlock> blocks = texts.getTextBlocks();
        if (blocks.size() == 0 || point == null) {
            mReadyForRecognition = true;
            return;
        }

        //convert the open cv point to graphics point
        android.graphics.Point penTopMidPoint = new android.graphics.Point((int)point.x , (int)point.y);

        double lowestDistance = 1000;
        String result = "";

        for (int i = 0; i < blocks.size(); i++) {
            List<FirebaseVisionText.Line> lines = blocks.get(i).getLines();
            for (int j = 0; j < lines.size(); j++) {
                List<FirebaseVisionText.Element> word = lines.get(j).getElements();
                for(int k = 0; k < word.size(); k++){
                    //result += (lines.get(j).getText() + " ");

                    //get the bottom mid point of the word
                   android.graphics.Point points[] = word.get(k).getCornerPoints();
                   android.graphics.Point wordBotMidPoint = getMidPoint(points[2] , points[3]);

                   //get the distance of the wordBotMidPoint from penTopMidPoint
                    double dist = getDistance(penTopMidPoint , wordBotMidPoint);

                    //update lowest distance if dist < lowestDistance
                    if(dist < lowestDistance) {
                        lowestDistance = dist;
                        result = word.get(k).getText().toLowerCase();
                    }

                }

            }

        }


        if(ttsInitResult == TextToSpeech.LANG_MISSING_DATA || ttsInitResult == TextToSpeech.LANG_NOT_SUPPORTED){
            showToast("Feauture not supported!");
        }else{
            Bundle params =  new Bundle();
            params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID,"");
            mTextToSpeech.speak(result,TextToSpeech.QUEUE_FLUSH,params,"UniqueID");
        }


    }


}
