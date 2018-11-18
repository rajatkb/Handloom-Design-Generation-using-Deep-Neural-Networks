import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.css']
})
export class MainComponent implements OnInit {

  ngOnInit() { }

  uploadedImage: File;
  uploadedImageURL: any;


  onImageSelect(event: any) {
    if (event.target.files && event.target.files[0]) {
      
      this.uploadedImage = event.target.files[0];
      let reader = new FileReader();
      reader.readAsDataURL(this.uploadedImage); 
      reader.onload = (event: any) => { 
        this.uploadedImageURL = event.target.result;
        
      }

    }
  }

  onImageDrop(event: any) {
    event.preventDefault();
    console.log(event);
  }

  allowImageDrop(event: any) {
    event.preventDefault();
  }

}
