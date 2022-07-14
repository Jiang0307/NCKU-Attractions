// Generated by view binder compiler. Do not edit!
package com.example.schoolspots.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.example.schoolspots.R;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import java.lang.NullPointerException;
import java.lang.Override;
import java.lang.String;

public final class ActivityCamBinding implements ViewBinding {
  @NonNull
  private final RelativeLayout rootView;

  @NonNull
  public final ImageView backimgbtn;

  @NonNull
  public final RelativeLayout camBarLayout;

  @NonNull
  public final FloatingActionButton camfab;

  @NonNull
  public final FloatingActionButton checkfab;

  @NonNull
  public final LinearLayout fabLayout;

  @NonNull
  public final ImageView img;

  @NonNull
  public final TextView messageText;

  @NonNull
  public final RelativeLayout progressLayout;

  @NonNull
  public final ProgressBar progressbar;

  @NonNull
  public final TextView titleText;

  @NonNull
  public final FloatingActionButton uploadfab;

  private ActivityCamBinding(@NonNull RelativeLayout rootView, @NonNull ImageView backimgbtn,
      @NonNull RelativeLayout camBarLayout, @NonNull FloatingActionButton camfab,
      @NonNull FloatingActionButton checkfab, @NonNull LinearLayout fabLayout,
      @NonNull ImageView img, @NonNull TextView messageText, @NonNull RelativeLayout progressLayout,
      @NonNull ProgressBar progressbar, @NonNull TextView titleText,
      @NonNull FloatingActionButton uploadfab) {
    this.rootView = rootView;
    this.backimgbtn = backimgbtn;
    this.camBarLayout = camBarLayout;
    this.camfab = camfab;
    this.checkfab = checkfab;
    this.fabLayout = fabLayout;
    this.img = img;
    this.messageText = messageText;
    this.progressLayout = progressLayout;
    this.progressbar = progressbar;
    this.titleText = titleText;
    this.uploadfab = uploadfab;
  }

  @Override
  @NonNull
  public RelativeLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static ActivityCamBinding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static ActivityCamBinding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.activity_cam, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static ActivityCamBinding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.backimgbtn;
      ImageView backimgbtn = ViewBindings.findChildViewById(rootView, id);
      if (backimgbtn == null) {
        break missingId;
      }

      id = R.id.camBar_layout;
      RelativeLayout camBarLayout = ViewBindings.findChildViewById(rootView, id);
      if (camBarLayout == null) {
        break missingId;
      }

      id = R.id.camfab;
      FloatingActionButton camfab = ViewBindings.findChildViewById(rootView, id);
      if (camfab == null) {
        break missingId;
      }

      id = R.id.checkfab;
      FloatingActionButton checkfab = ViewBindings.findChildViewById(rootView, id);
      if (checkfab == null) {
        break missingId;
      }

      id = R.id.fab_layout;
      LinearLayout fabLayout = ViewBindings.findChildViewById(rootView, id);
      if (fabLayout == null) {
        break missingId;
      }

      id = R.id.img;
      ImageView img = ViewBindings.findChildViewById(rootView, id);
      if (img == null) {
        break missingId;
      }

      id = R.id.message_text;
      TextView messageText = ViewBindings.findChildViewById(rootView, id);
      if (messageText == null) {
        break missingId;
      }

      id = R.id.progress_layout;
      RelativeLayout progressLayout = ViewBindings.findChildViewById(rootView, id);
      if (progressLayout == null) {
        break missingId;
      }

      id = R.id.progressbar;
      ProgressBar progressbar = ViewBindings.findChildViewById(rootView, id);
      if (progressbar == null) {
        break missingId;
      }

      id = R.id.title_text;
      TextView titleText = ViewBindings.findChildViewById(rootView, id);
      if (titleText == null) {
        break missingId;
      }

      id = R.id.uploadfab;
      FloatingActionButton uploadfab = ViewBindings.findChildViewById(rootView, id);
      if (uploadfab == null) {
        break missingId;
      }

      return new ActivityCamBinding((RelativeLayout) rootView, backimgbtn, camBarLayout, camfab,
          checkfab, fabLayout, img, messageText, progressLayout, progressbar, titleText, uploadfab);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}