package com.scliang.x.media;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager.widget.PagerAdapter;
import androidx.viewpager.widget.ViewPager;

import java.io.File;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private final String[] C01 = new String[] { "0","1" };
    private final String[] C10 = new String[] { "1","0" };
    private int camMerge = 0;
    private String[] cs = C01;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        MediaManager.init(this, findViewById(R.id.gl_view));
        if (hasPermissions()) {
            setupPermissionView(true);
        } else {
            requestPermissions(new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.READ_EXTERNAL_STORAGE}, 1001);
            setupPermissionView(false);
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        if (hasPermissions()) {
            MediaManager.start();
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (hasPermissions()) {
            MediaManager.stop();
        }
    }

    @Override
    protected void onDestroy() {
        if (hasPermissions()) {
            MediaManager.release();
        }
        super.onDestroy();
    }

    private boolean hasPermissions() {
        return PackageManager.PERMISSION_GRANTED == checkSelfPermission(Manifest.permission.CAMERA) &&
                PackageManager.PERMISSION_GRANTED == checkSelfPermission(Manifest.permission.RECORD_AUDIO) &&
                PackageManager.PERMISSION_GRANTED == checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1001 &&
                PackageManager.PERMISSION_GRANTED == grantResults[0] &&
                PackageManager.PERMISSION_GRANTED == grantResults[1] &&
                PackageManager.PERMISSION_GRANTED == grantResults[2]) {
            setupPermissionView(true);
            MediaManager.start();
        }
    }


    /*
     *
     */
    private void setupPermissionView(boolean hasPermission) {
        View noPermission = findViewById(R.id.no_permissions);
        if (noPermission != null) {
            noPermission.setVisibility(hasPermission?View.GONE:View.VISIBLE);
        }
        View recordContainer = findViewById(R.id.record_container);
        if (recordContainer != null) {
            recordContainer.setVisibility(hasPermission?View.VISIBLE:View.GONE);
        }
        View playContainer = findViewById(R.id.play_container);
        if (playContainer != null) {
            playContainer.setVisibility(hasPermission?View.VISIBLE:View.GONE);
        }
        ViewPager viewPager = findViewById(R.id.view_pager);
        if (viewPager != null) {
            viewPager.setVisibility(hasPermission?View.VISIBLE:View.GONE);
            viewPager.setAdapter(new TypeAdapter());
            viewPager.clearOnPageChangeListeners();
            viewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() {
                @Override
                public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) { }

                @Override
                public void onPageSelected(int position) {
                    checkType(position);
                }

                @Override
                public void onPageScrollStateChanged(int state) { }
            });
        }
        Button effect = findViewById(R.id.effect);
        if (effect != null) effect.setVisibility(hasPermission?View.VISIBLE:View.GONE);
        if (effect != null) effect.setOnClickListener(v -> {
            List<String> names = MediaManager.getSupportedEffectPaints();
            String[] items = new String[names.size()];
            for (int i = 0; i < names.size(); i++) {
                String name = names.get(i);
                items[i] = name;
            }
            AlertDialog.Builder listDialog = new AlertDialog.Builder(this);
            listDialog.setItems(items, (dialog, which) -> {
                String name = items[which];
                effect.setText(name);
                MediaManager.updateEffectPaint(name);
            });
            listDialog.show();
        });
        setupRecordViews(hasPermission);
        setupPlayViews(hasPermission);
        checkType(0);
    }

    private void setupRecordViews(boolean hasPermission) {
        Button camera = findViewById(R.id.camera);
        Button record = findViewById(R.id.record);
        Button merge = findViewById(R.id.merge);
        Button awb = findViewById(R.id.awb);
        if (hasPermission) {
            if (camera != null) camera.setVisibility(View.VISIBLE);
            if (camera != null) camera.setOnClickListener(v -> {
                if (cs == C01) cs = C10;
                else if (cs == C10) cs = C01;
                MediaManager.preview(camMerge, cs);
                if (awb != null) awb.setText("AUTO");
            });
            if (record != null) record.setVisibility(View.VISIBLE);
            if (record != null) record.setOnClickListener(v -> {
                if (MediaManager.recording()) MediaManager.stopRecord();
                else MediaManager.startRecord(getFilesDir() + "/demo.mp4");
            });
            if (merge != null) merge.setVisibility(View.VISIBLE);
            if (merge != null) merge.setOnClickListener(v -> {
                String[] items = new String[] { "SINGLE", "VERTICAL", "CHAT" };
                AlertDialog.Builder listDialog = new AlertDialog.Builder(this);
                listDialog.setItems(items, (dialog, which) -> {
                    String name = items[which];
                    merge.setText(name);
                    camMerge = which;
                    MediaManager.preview(camMerge, cs);
                });
                listDialog.show();
            });
            if (awb != null) awb.setVisibility(View.VISIBLE);
            if (awb != null) awb.setOnClickListener(v -> {
                String[] items = new String[] { "AUTO", "INCANDESCENT", "FLUORESCENT", "DAYLIGHT", "CLOUDY_DAYLIGHT" };
                AlertDialog.Builder listDialog = new AlertDialog.Builder(this);
                listDialog.setItems(items, (dialog, which) -> {
                    String name = items[which];
                    int val;
                    switch (name) {
                        default:
                        case "AUTO":
                            val = 1;
                            break;
                        case "INCANDESCENT":
                            val = 2;
                            break;
                        case "FLUORESCENT":
                            val = 3;
                            break;
                        case "DAYLIGHT":
                            val = 5;
                            break;
                        case "CLOUDY_DAYLIGHT":
                            val = 6;
                            break;
                    }
                    MediaManager.setCameraAWB(cs[0], val, (id, value, result) -> {
                        if (result) awb.setText(name);
                    });
                });
                listDialog.show();
            });
        }
    }

    private void setupPlayViews(boolean hasPermission) {
        Button play = findViewById(R.id.play);
        if (hasPermission) {
            if (play != null) play.setVisibility(View.VISIBLE);
            if (play != null) play.setOnClickListener(v -> {
                if (MediaManager.playing()) {
                    MediaManager.stopPlay();
                }
                else {
                    MediaManager.startPlay(getDemoMp4());
                }
            });
        }
    }

    private String getDemoMp4() {
//        return getFilesDir() + "/demo.mp4";
        return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                .getAbsolutePath() + "/demo.mp4";
    }

    private void checkType(int position) {
        View recordContainer = findViewById(R.id.record_container);
        View playContainer = findViewById(R.id.play_container);
        if (position == 0) {
            if (recordContainer != null) {
                recordContainer.setVisibility(View.VISIBLE);
            }
            if (playContainer != null) {
                playContainer.setVisibility(View.GONE);
            }
            if (hasPermissions()) {
                MediaManager.stopPlay();
                MediaManager.preview(camMerge, cs);
            }
        } else {
            if (recordContainer != null) {
                recordContainer.setVisibility(View.GONE);
            }
            if (playContainer != null) {
                playContainer.setVisibility(View.VISIBLE);
            }
            if (hasPermissions()) {
                MediaManager.stopPreview();
                MediaManager.startPlay(getDemoMp4());
            }
        }
    }

    private static class TypeAdapter extends PagerAdapter {

        @Override
        public int getCount() {
            return 2;
        }

        @Override
        public boolean isViewFromObject(@NonNull View view, @NonNull Object object) {
            return view == object;
        }

        @NonNull
        @Override
        public Object instantiateItem(@NonNull ViewGroup container, int position) {
            TextView textView = new TextView(container.getContext());
            textView.setGravity(Gravity.CENTER);
            container.addView(textView);
            return textView;
        }

        @Override
        public void destroyItem(@NonNull ViewGroup container, int position, @NonNull Object object) {
            container.removeView((View) object);
        }
    }
}
