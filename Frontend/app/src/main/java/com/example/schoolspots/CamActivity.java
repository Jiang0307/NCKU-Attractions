package com.example.schoolspots;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.animation.OvershootInterpolator;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.resource.bitmap.RoundedCorners;
import com.bumptech.glide.request.RequestOptions;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.storage.FileDownloadTask;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class CamActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    ActivityResultLauncher<Intent> activityResultLauncher;
    ActivityResultLauncher<Uri> uriActivityResultLauncher;

    TextView titleTv, messageTv;
    ImageView img, backimgBtn;

    FloatingActionButton cam_fab, openImgFile_Fab, upload_Fab, check;
    Boolean Menustate = false;
    int translationy = 100;

    ProgressBar progressBar;
    RelativeLayout layout;

    private StorageReference storageReference;
    private DatabaseReference databaseReference;

    private String currentPhotoPath;
    Uri photoURI = null;

    //動畫
    OvershootInterpolator overshootInterpolator = new OvershootInterpolator();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cam);

        init();
    }

    private void init() {
        cam_fab = findViewById(R.id.camfab);
        //openImgFile_Fab = findViewById(R.id.filefab);
        upload_Fab = findViewById(R.id.uploadfab);
        check = findViewById(R.id.checkfab);

        layout = findViewById(R.id.progress_layout);
        progressBar = findViewById(R.id.progressbar);
        img = findViewById(R.id.img);
        titleTv = findViewById(R.id.title_text);
        messageTv = findViewById(R.id.message_text);

        backimgBtn = findViewById(R.id.backimgbtn);

        //進度條
        progressBar.setVisibility(View.GONE);

        cam_fab.setTranslationY(translationy);
        cam_fab.animate().alpha(0f).setDuration(100).start();
//        openImgFile_Fab.setTranslationY(translationy);
//        openImgFile_Fab.animate().alpha(0f).setDuration(100).start();
        upload_Fab.setTranslationY(translationy);
        upload_Fab.animate().alpha(0f).setDuration(100).start();

        check.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Menustate) {
                    closeMenu();
                } else {
                    openMenu();
                }
            }
        });

        cam_fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dispatchTakePictureIntent_OpenCam();
            }
        });

//        openImgFile_Fab.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//
//            }
//        });

        //檢查路徑是否為空，上傳照片
        upload_Fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                if (photoURI != null && img.getDrawable() != null) {
                uploadPicture();
                progressBar.setVisibility(View.VISIBLE);
//                } else {
//
//                }
            }
        });

        //關閉頁面
        backimgBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });

//        activityResultLauncher = registerForActivityResult(
//                new ActivityResultContracts.StartActivityForResult(),
//                new ActivityResultCallback<ActivityResult>() {
//                    @Override
//                    public void onActivityResult(ActivityResult result) {
//                        if (result.getResultCode() == RESULT_OK && result.getData() != null) {
//
//                            //儲存透過路徑取得照片
//                            File takePitcureFromFile = new File(currentPhotoPath);
//                            img.setImageURI(Uri.fromFile(takePitcureFromFile));
//                            Log.e("TAG", takePitcureFromFile.toString());
//                        } else {
//                            finish();
//                        }
//                    }
//                });

        uriActivityResultLauncher = registerForActivityResult(new ActivityResultContracts.TakePicture(), new ActivityResultCallback<Boolean>() {
            @Override
            public void onActivityResult(Boolean result) {

                img.setImageURI(photoURI);
            }
        });

        dispatchTakePictureIntent_OpenCam();

        //檢查狀態是否為2 修改狀態回到0
        databaseReference = FirebaseDatabase.getInstance().getReference("pictureStatus");
        databaseReference.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.getValue(int.class) == 2) {
                    uploadPictureStatus(0);
                    getPicResultText();
                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
            }
        });
    }

    //建立檔案名稱
    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";

        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    //存檔
    private void dispatchTakePictureIntent_OpenCam() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                // Error occurred while creating the File
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {
                photoURI = FileProvider.getUriForFile(this,
                        "com.example.schoolspots.fileprovider",
                        photoFile);
                uriActivityResultLauncher.launch(photoURI);
                titleTv.setText("");
                messageTv.setText("");
            }
        }
    }

    //修改資料庫狀態
    private void uploadPictureStatus(int status) {
        databaseReference = FirebaseDatabase.getInstance().getReference("pictureStatus");
        databaseReference.setValue(status);
    }

    //上傳相片(test.jpg),成功後修改狀態值
    private void uploadPicture() {
        storageReference = FirebaseStorage.getInstance().getReference().child("test.jpg");
        storageReference.putFile(photoURI)
                .addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                    @Override
                    public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                        Toast.makeText(CamActivity.this, "上傳成功,等待載入...", Toast.LENGTH_SHORT).show();
                        uploadPictureStatus(1);
                        img.setVisibility(View.INVISIBLE);
                        titleTv.setText("");
                        messageTv.setText("");
                    }
                }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
                Toast.makeText(CamActivity.this, "上傳失敗", Toast.LENGTH_SHORT).show();
            }
        });
    }

    //若辨識成功 執行取得相片辨識結果 修改狀態 並下載景點圖片
    private void getPicResultText() {
        databaseReference = FirebaseDatabase.getInstance().getReference("result");
        databaseReference.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists()) {
                    useResultFindData(snapshot.getValue().toString());
//                    retrieveImage(snapshot.getValue().toString());
                    loadImageWithUrl(snapshot.getValue().toString());
                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                Toast.makeText(CamActivity.this, "下載圖片失敗", Toast.LENGTH_SHORT).show();
            }
        });
    }

    /**
     * 判断Activity是否Destroy
     *
     * @param
     * @return
     */
    public static boolean isDestroy(Activity mActivity) {
        if (mActivity == null || mActivity.isFinishing() || (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR1 && mActivity.isDestroyed())) {
            return true;
        } else {
            return false;
        }
    }

    //下載圖片(pics/地點)
    private void retrieveImage(String result) {
        storageReference = FirebaseStorage.getInstance().getReference().child("pics/" + result + ".jpg");

        try {
            File localfile = File.createTempFile("tempfile", ".jpg");

            storageReference.getFile(localfile)
                    .addOnSuccessListener(new OnSuccessListener<FileDownloadTask.TaskSnapshot>() {
                        @Override
                        public void onSuccess(FileDownloadTask.TaskSnapshot taskSnapshot) {
                            Bitmap bitmap = BitmapFactory.decodeFile(localfile.getAbsolutePath());
                            img.setImageBitmap(bitmap);

                            float alpha = (float) 0.1;
                            titleTv.setAlpha(alpha);

                            img.setVisibility(View.VISIBLE);
                            img.animate().alpha(1).start();
                        }
                    }).addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception e) {

                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadImageWithUrl(String attractionChild) {
        databaseReference = FirebaseDatabase.getInstance().getReference("spots").child(attractionChild);
        databaseReference.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (!isDestroy((Activity) CamActivity.this)) {
                    Glide.with(CamActivity.this).load(snapshot.child("pic_url").getValue().toString()).into(img);
                }
//                Glide.with(img.getContext()).load(snapshot.child("pic_url").getValue().toString()).into(img);

                float alpha = (float) 0.1;
                titleTv.setAlpha(alpha);

                img.setVisibility(View.VISIBLE);
                img.animate().alpha(1).start();
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });
    }

    //透過辨識結果 取得景點介紹
    private void useResultFindData(String result) {
        databaseReference = FirebaseDatabase.getInstance().getReference("spots").child(result);
        databaseReference.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                String message = snapshot.child("introduce").getValue().toString();

                progressBar.setVisibility(View.GONE);

                messageTv.setText(message);
                titleTv.setText(result);

                titleTv.setVisibility(View.VISIBLE);
                messageTv.setVisibility(View.VISIBLE);

                viewAnimation();
                if (Menustate) {
                    closeMenu();
                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });
    }

    //動畫設定
    private void viewAnimation() {
        titleTv.setTranslationY(300);
        messageTv.setTranslationY(300);

        float alpha = (float) 0.1;
        titleTv.setAlpha(alpha);
        messageTv.setAlpha(alpha);
        titleTv.animate().translationY(0).alpha(1).setDuration(700).setStartDelay(300).start();
        messageTv.animate().translationY(0).alpha(1).setDuration(700).setStartDelay(300).start();
    }

    //功能按鈕清單動畫
    private void openMenu() {
        Menustate = !Menustate;

        check.animate().translationY(0f).setInterpolator(overshootInterpolator).rotation(45).setDuration(300).start();
        cam_fab.animate().translationY(0f).alpha(1f).setInterpolator(overshootInterpolator).setDuration(300).start();
        cam_fab.setVisibility(View.VISIBLE);
//        openImgFile_Fab.animate().translationY(0f).alpha(1f).setInterpolator(overshootInterpolator).setDuration(300).start();
//        openImgFile_Fab.setVisibility(View.VISIBLE);
        upload_Fab.animate().translationY(0f).alpha(1f).setInterpolator(overshootInterpolator).setDuration(300).start();
        upload_Fab.setVisibility(View.VISIBLE);

    }

    private void closeMenu() {
        Menustate = !Menustate;
        check.animate().translationY(0f).setInterpolator(overshootInterpolator).rotation(0).setDuration(300).start();
        cam_fab.animate().translationY(translationy).alpha(0f).setInterpolator(overshootInterpolator).setDuration(300).start();
        cam_fab.setVisibility(View.VISIBLE);
//        openImgFile_Fab.animate().translationY(translationy).alpha(0f).setInterpolator(overshootInterpolator).setDuration(300).start();
//        openImgFile_Fab.setVisibility(View.INVISIBLE);
        upload_Fab.animate().translationY(translationy).alpha(0f).setInterpolator(overshootInterpolator).setDuration(300).start();
        upload_Fab.setVisibility(View.INVISIBLE);
    }
}