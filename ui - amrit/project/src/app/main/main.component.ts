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
  icon1Hovered:boolean = false;
  icon2Hovered:boolean = false;
  icon3Hovered:boolean = false;
  icon4Hovered:boolean = false;
  imageSelected:boolean = false;

  onImageSelect(event: any) {
    if (event.target.files && event.target.files[0]) {
      
      this.uploadedImage = event.target.files[0];
      let reader = new FileReader();
      reader.readAsDataURL(this.uploadedImage); 
      reader.onload = (event: any) => { 
        this.uploadedImageURL = event.target.result;
        this.imageSelected = true;
        console.log(this.imageSelected);
      }

    }
  }

  onIconHover(event: any) {
    //this.iconActive = true;
    let hoveredIcon = event.target.classList[2];
    if(hoveredIcon == "fa-picture-o")
      this.icon1Hovered = true;
    else if(hoveredIcon == "fa-file-code-o")
      this.icon2Hovered = true;
    else if(hoveredIcon == "fa-paint-brush")
      this.icon3Hovered = true;
    else if(hoveredIcon == "fa-pencil-square-o")
      this.icon4Hovered = true;
  }

  onIconLeave(event: any) {
    let leftIcon = event.target.classList[2];
    if(leftIcon == "fa-picture-o")
      this.icon1Hovered = false;
    else if(leftIcon == "fa-file-code-o")
      this.icon2Hovered = false;
    else if(leftIcon == "fa-paint-brush")
      this.icon3Hovered = false;
    else if(leftIcon == "fa-pencil-square-o")
      this.icon4Hovered = false;
  }

  onImageDrop(event: any) {
    event.preventDefault();
    console.log(event);
  }

  allowImageDrop(event: any) {
    event.preventDefault();
  }

}
