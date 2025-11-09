package com.example.schoolspots;

import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.firebase.ui.database.FirebaseRecyclerAdapter;
import com.firebase.ui.database.FirebaseRecyclerOptions;

public class adapter extends FirebaseRecyclerAdapter<attraction_model, adapter.viewholder> {
    Context context;

    public adapter(@NonNull FirebaseRecyclerOptions<attraction_model> options, Context context) {
        super(options);
        this.context = context;
    }

    @Override
    protected void onBindViewHolder(@NonNull viewholder holder, int position, @NonNull attraction_model model) {
        holder.name.setText(getRef(position).getKey());
        holder.introduce.setText(model.getIntroduce());

        Glide.with(holder.img.getContext()).load(model.getPic_url()).into(holder.img);

        holder.img.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v) {
                Intent introduceIntent = new Intent(context, IntroduceActivity.class);
                introduceIntent.putExtra("getIntroduceKey", getRef(position).getKey());

                context.startActivity(introduceIntent);
            }
        });
    }

    @NonNull
    @Override
    public viewholder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.attraction_list, parent, false);
        return new viewholder(view);
    }

    class viewholder extends RecyclerView.ViewHolder {

        ImageView img;
        TextView name, introduce;

        public viewholder(@NonNull View itemView) {
            super(itemView);

            img = itemView.findViewById(R.id.attraction_img);
            name = itemView.findViewById(R.id.attraction_name);
            introduce = itemView.findViewById(R.id.attraction_info);
        }
    }

    public interface RecyclerviewListener {
        void onCLick(View v, int position);
    }
}
