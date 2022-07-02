package com.example.schoolspots;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;

import com.firebase.ui.database.FirebaseRecyclerOptions;
import com.google.firebase.database.FirebaseDatabase;

public class AttractionActivity extends AppCompatActivity {

    RecyclerView recyclerView;
    adapter adapter;
    ImageView attraction_backImg_Btn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_attraction);

        recyclerView = findViewById(R.id.attraction_recycler);

        attraction_backImg_Btn = findViewById(R.id.attraction_backimgbtn);

        attraction_backImg_Btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });

        GridLayoutManager layoutManager = new GridLayoutManager(this, 2);
        recyclerView.setLayoutManager(layoutManager);

        FirebaseRecyclerOptions<attraction_model> options =
                new FirebaseRecyclerOptions.Builder<attraction_model>()
                        .setQuery(FirebaseDatabase.getInstance().getReference().child("spots"), attraction_model.class)
                        .build();

        adapter = new adapter(options, this);

        recyclerView.setAdapter(adapter);
        anim();
    }

    @Override
    protected void onStart() {
        super.onStart();

        adapter.startListening();
    }

    @Override
    protected void onStop() {
        super.onStop();

        adapter.stopListening();
    }


    private void anim() {
        recyclerView.setTranslationY(900);

        float alpha = (float) 0.1;
        recyclerView.setAlpha(alpha);
        recyclerView.animate().translationY(0).alpha(1).setDuration(900).setStartDelay(300).start();
    }
}