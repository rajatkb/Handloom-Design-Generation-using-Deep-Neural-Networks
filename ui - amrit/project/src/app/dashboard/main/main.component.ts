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
  icon6Hovered:boolean = false;
  imageSelected:boolean = false;

  gallerySelected:boolean = false;

  constructor(public router:Router,	public authService:AuthService,	public flashMessagesService:FlashMessagesService,
    public uploadService:UploadService, public storage: AngularFireStorage, public db: AngularFirestore, 
    public afdb: AngularFireDatabase, public afAuth: AngularFireAuth) { }

  ngOnInit() { 

  }

  onLogout() {
    this.authService.logout();
  }

  toggleHover(event: boolean) {
    this.isHovering = event;
  }

  startUpload(event: FileList) {
    const file = event.item(0);
    this.uploadedImage = file;

    if(file.type.split('/')[0] != 'image') {
      console.log("Unsupported file format");
    }

    const filename = `${new Date().getTime()}_${file.name}`;
    
    this.task = this.storage.upload('/images/' + this.afAuth.auth.currentUser.uid + '/' + filename, file);
    this.afdb.list('/users/'+ this.afAuth.auth.currentUser.uid + '/images').push(filename);

    let reader = new FileReader();
    reader.readAsDataURL(this.uploadedImage); 
    reader.onload = (event: any) => { 
      this.uploadedImageURL = event.target.result;
      this.imageSelected = true;
    }

  }

  isActive(snapshot: any) {
    return snapshot.state === 'running' && snapshot.bytesTransferred < snapshot.totalBytes;
  }

  onGallerySelect(event: Event) {
    this.gallerySelected = true;
  }

  onImageSelect(event: any) {
    if (event.target.files && event.target.files[0]) {
      
      this.uploadedImage = event.target.files[0];
      this.startUpload(event.target.files);

      
      this.uploadService.upload(this.uploadedImage).subscribe((res:Object) => {
          let bytestring = res['status'];
          let image = bytestring.split('\'')[1];
          this.uploadedImageURL = 'data:image/jpeg;base64,' + image;
        }, error => {
          if(error) {
            this.flashMessagesService.show("Communication failed",{cssClass: 'custom-danger-alert' , timeOut:7000});
            console.log(error);
          }
      });
    }

  }

  onIconHover(event: any) {
    //this.iconActive = true;
    let hoveredIcon = event.target.classList[2];
    if(hoveredIcon == "fa-file-image-o")
      this.icon1Hovered = true;
    else if(hoveredIcon == "fa-file-code-o")
      this.icon2Hovered = true;
    else if(hoveredIcon == "fa-paint-brush")
      this.icon3Hovered = true;
    else if(hoveredIcon == "fa-picture-o")
      this.icon4Hovered = true;
    else if(hoveredIcon == "fa-sign-out")
      this.icon5Hovered = true;
    else if(hoveredIcon == "fa-home")
      this.icon6Hovered = true;
  }

  onIconLeave(event: any) {
    let leftIcon = event.target.classList[2];
    if(leftIcon == "fa-file-image-o")
      this.icon1Hovered = false;
    else if(leftIcon == "fa-file-code-o")
      this.icon2Hovered = false;
    else if(leftIcon == "fa-paint-brush")
      this.icon3Hovered = false;
    else if(leftIcon == "fa-picture-o")
      this.icon4Hovered = false;
    else if(leftIcon == "fa-sign-out")
      this.icon5Hovered = false;
    else if(leftIcon == "fa-home")
      this.icon6Hovered = false;
  }

  // onImageDrop(event: any) {
  //   event.preventDefault();
  //   console.log(event);
  // }

  // allowImageDrop(event: any) {
  //   event.preventDefault();
  // }


}
