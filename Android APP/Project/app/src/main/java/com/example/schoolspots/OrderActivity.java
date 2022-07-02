package com.example.schoolspots;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.transition.ChangeBounds;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

public class OrderActivity extends AppCompatActivity {

    ImageView logo_image;
    TextView titletv;
    LinearLayout main_order;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_order);
        init();
    }

    private void init() {
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
        getWindow().setEnterTransition(null);

        logo_image = findViewById(R.id.logo_img);
        titletv = findViewById(R.id.main_title_tv);
        main_order = findViewById(R.id.main_order);

        getWindow().setSharedElementEnterTransition(new ChangeBounds().setDuration(1200));

        viewAnimation();
    }

    //開啟相機函式 -> 詢問權限
    public void OpenCam(View view) {
        askCameraPermissions();
    }

    //詢問相機權限
    private void askCameraPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 101);
        } else {
            intoCamPage();
        }
    }

    //權限結果
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 101) {
            if (grantResults.length < 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                intoCamPage();
            } else {
                Toast.makeText(this, "未開啟相機權限", Toast.LENGTH_SHORT).show();
            }
        }
    }

    //進入辨識頁面
    private void intoCamPage() {
        Intent camIntent = new Intent(this, CamActivity.class);
        startActivity(camIntent);
    }

    //進入景點介紹
    public void AttractionInfo(View view) {
        Intent attractionIntent = new Intent(this, AttractionActivity.class);
        startActivity(attractionIntent);
    }

    //動畫設定
    private void viewAnimation() {
        main_order.setTranslationY(900);

        float alpha = (float) 0.1;
        main_order.setAlpha(alpha);
        main_order.animate().translationY(0).alpha(1).setDuration(1100).setStartDelay(300).start();
    }

    //呼叫動畫
    public void anim(View view) {
        viewAnimation();
    }

}