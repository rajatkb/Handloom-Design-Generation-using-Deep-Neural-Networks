import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { UploadService } from '../../services/upload.service';
import { FlashMessagesService } from 'angular2-flash-messages';
import { AngularFireStorage, AngularFireUploadTask } from 'angularfire2/storage';
import { AngularFirestore } from 'angularfire2/firestore';
import { Observable } from 'rxjs';
import { tap, map } from 'rxjs/operators';
import { AngularFireDatabase } from 'angularfire2/database';
import { AngularFireAuth } from 'angularfire2/auth';
import { DataService } from '../../services/data.service'

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.css']
})
export class MainComponent implements OnInit {

  task: AngularFireUploadTask;
  snapshot: Observable<any>;
  downloadURL: Observable<string>;
  isHovering: boolean;
  percentage: Observable<any>;

  uploadedImage: File;
  uploadedImageURL: any;
  icon1Hovered:boolean = false;
  icon2Hovered:boolean = false;
  icon3Hovered:boolean = false;
  icon4Hovered:boolean = false;
  icon5Hovered:boolean = false;
  imageSelected:boolean = false;
  maskButtonCaption:string = "View Mask";

  gallerySelected:boolean = false;
  isMaskActive:boolean = false;
  
  constructor(public router:Router,	public authService:AuthService,	public flashMessagesService:FlashMessagesService,
    public uploadService:UploadService, public storage: AngularFireStorage, public db: AngularFirestore, 
    public afdb: AngularFireDatabase, public afAuth: AngularFireAuth, public dataService: DataService) { }

  ngOnInit() {
    this.dataService.currentMessage.subscribe(obj => this.uploadedImageURL = obj["imageSrc"]);
  }

  onLogout() {
    this.authService.logout();
  }

  toggleHover(event: boolean) {
    this.isHovering = event;
  }

  uploadToFirebase() {
    let file = this.uploadedImage;
    //this.uploadedImage = file;

    // if(file.type.split('/')[0] != 'image') {
    //   console.log("Unsupported file format");
    // }

    const filename = `${new Date().getTime()}_${file.name}`;
    
    this.task = this.storage.upload('/images/' + this.afAuth.auth.currentUser.uid + '/' + filename, file);
    this.afdb.list('/users/'+ this.afAuth.auth.currentUser.uid + '/images').push(filename);

    // let reader = new FileReader();
    // reader.readAsDataURL(this.uploadedImage); 
    // reader.onload = (event: any) => { 
    //   this.uploadedImageURL = event.target.result;
    //   this.imageSelected = true;
    // }

  }

  isActive(snapshot: any) {
    return snapshot.state === 'running' && snapshot.bytesTransferred < snapshot.totalBytes;
  }

  onImageSelect(event: any) {


    if(event.type == undefined) {
      this.uploadedImage = event[0];
    } else {
      this.uploadedImage = event.target.files[0];
    }

    let reader = new FileReader();
    reader.readAsDataURL(this.uploadedImage); 
    reader.onload = (event: any) => { 
      this.uploadedImageURL = event.target.result;
      this.imageSelected = true;
      this.dataService.changeMessage({"file": this.uploadedImage, "imageSrc": this.uploadedImageURL, "isMaskActive": this.isMaskActive});
    }

  }

  selectedGANIcon(event: Event) {

    if(this.gallerySelected == true) {
      this.flashMessagesService.show("Please switch back to original image from gallery",{cssClass: 'custom-danger-alert' , timeOut:7000});
    }
    else if(this.uploadedImage == undefined) {
      this.flashMessagesService.show("Please provide an image first",{cssClass: 'custom-danger-alert' , timeOut:7000});
    }
    else if(this.isMaskActive == true) {
      this.flashMessagesService.show("Please switch back to the original image from mask",{cssClass: 'custom-danger-alert' , timeOut:7000});
    }
    else {
      this.uploadService.uploadForGAN(this.uploadedImage).subscribe((res:Object) => {
          let bytestring = res['status'];
          let image = bytestring.split('\'')[1];
          //console.log(image);
          this.uploadedImageURL = 'data:image/jpeg;base64,' + image;
        }, error => {
          if(error) {
            this.flashMessagesService.show("Communication with the server failed",{cssClass: 'custom-danger-alert' , timeOut:7000});
            console.log(error);
          }
      });
    }
  
  }

  selectedMaskIcon(event: Event) {

    if(this.gallerySelected == true) {
      this.flashMessagesService.show("Please switch back to original image from gallery",{cssClass: 'custom-danger-alert' , timeOut:7000});
    }
    else if(this.uploadedImage == undefined) {
      this.flashMessagesService.show("Please provide an image first",{cssClass: 'custom-danger-alert' , timeOut:7000});
    }
    else {
      this.isMaskActive = !(this.isMaskActive);
      this.dataService.changeMessage({"file": this.uploadedImage, "imageSrc": this.uploadedImageURL, "isMaskActive": this.isMaskActive});

      if(this.isMaskActive) {

        this.maskButtonCaption = "Reset to original";

        this.uploadService.uploadForMask(this.uploadedImage).subscribe((res:Object) => {
          console.log("log: receiving response for mask")  
          let bytestring = res['status'];
            let image = bytestring.split('\'')[1];
            //console.log(image);
            this.uploadedImageURL = 'data:image/jpeg;base64,' + image;
          }, error => {
            if(error) {
              this.flashMessagesService.show("Communication with the server failed",{cssClass: 'custom-danger-alert' , timeOut:7000});
              console.log(error);
            }
        });

      } 

      else {
        this.maskButtonCaption = "View Mask";
        let reader = new FileReader();
        reader.readAsDataURL(this.uploadedImage); 
        reader.onload = (event: any) => { 
          this.uploadedImageURL = event.target.result;
          this.imageSelected = true;
          this.dataService.changeMessage({"file": this.uploadedImage, "imageSrc": this.uploadedImageURL, "isMaskActive": this.isMaskActive});
        }
      }

    }

  }

  onIconHover(event: any) {
    let hoveredIcon = event.target.classList[2];
    if(hoveredIcon == "fa-picture-o")
      this.icon1Hovered = true;
    else if(hoveredIcon == "fa-home")
      this.icon2Hovered = true;
    else if(hoveredIcon == "fa-paint-brush")
      this.icon3Hovered = true;
    else if(hoveredIcon == "special-class")
      this.icon4Hovered = true;
    else if(hoveredIcon == "fa-sign-out")
      this.icon5Hovered = true;
  }

  onIconLeave(event: any) {
    let leftIcon = event.target.classList[2];
    
    if(leftIcon == "fa-picture-o")
      this.icon1Hovered = false;
    else if(leftIcon == "fa-home")
      this.icon2Hovered = false;
    else if(leftIcon == "fa-paint-brush")
      this.icon3Hovered = false;
    else if(leftIcon == "special-class")
      this.icon4Hovered = false;
    else if(leftIcon == "fa-sign-out")
      this.icon5Hovered = false;
  }

}
