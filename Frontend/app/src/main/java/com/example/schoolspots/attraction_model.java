package com.example.schoolspots;

public class attraction_model {
    String introduce, pic_url, name;

    attraction_model() {

    }

    public attraction_model(String introduce, String pic_url, String name) {
        this.introduce = introduce;
        this.pic_url = pic_url;
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getIntroduce() {
        return introduce;
    }

    public void setIntroduce(String introduce) {
        this.introduce = introduce;
    }

    public String getPic_url() {
        return pic_url;
    }

    public void setPic_url(String pic_url) {
        this.pic_url = pic_url;
    }
}
