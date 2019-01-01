import { Component, OnInit } from '@angular/core';
import { DataService } from '../../services/data.service';
import { FlashMessagesService } from 'angular2-flash-messages';
import { UploadService } from '../../services/upload.service';

@Component({
  selector: 'app-landing',
  templateUrl: './landing.component.html',
  styleUrls: ['./landing.component.css']
})
export class LandingComponent implements OnInit {

  chosenForeground:string = "";
  chosenBackground:string = "";
  uploadedImage:File;
  isMaskActive:boolean;

  constructor(public dataService: DataService, public flashMessagesService:FlashMessagesService, public uploadService:UploadService) { }

  ngOnInit() {
    this.dataService.currentMessage.subscribe(obj => {this.uploadedImage = obj["file"]; this.isMaskActive = obj["isMaskActive"];});
  }

  foregroundSelected(event: any) {
    this.chosenForeground = event.target.id;
  }

  backgroundSelected(event: any) {
    this.chosenBackground = event.target.id;
  }

  uploadForMerge() {
    if(this.chosenBackground.length == 0 || this.chosenForeground.length == 0 || this.uploadedImage == undefined || this.uploadedImage.size == undefined) {
      this.flashMessagesService.show("Please ensure that all three parameters are chosen",{cssClass: 'custom-danger-alert' , timeOut:7000});
    }
    else if(this.isMaskActive == true) {
      this.flashMessagesService.show("Please switch back to the original image from mask",{cssClass: 'custom-danger-alert' , timeOut:7000});
    }
    else {
      this.uploadService.uploadForMerge(this.uploadedImage, this.chosenBackground, this.chosenForeground).subscribe((res:Object) => {
          console.log("log: results recieved , loading results");
          let bytestring = res['status'];
          let image = bytestring.split('\'')[1];
          //console.log(image);
          let url = 'data:image/jpeg;base64,' + image;
          this.dataService.changeMessage({"file": this.uploadedImage, "imageSrc": url, "isMaskActive": this.isMaskActive});
          //this.dataService.currentMessage.subscribe(obj => console.log(obj));
        }, error => {
          if(error) {
            this.flashMessagesService.show("Communication with the server failed",{cssClass: 'custom-danger-alert' , timeOut:7000});
            console.log(error);
          }
      });
    }
  }


}
