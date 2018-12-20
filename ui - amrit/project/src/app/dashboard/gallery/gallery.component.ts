import { Component, OnInit } from '@angular/core';
import { AngularFireAuth } from 'angularfire2/auth';
import { AuthService } from '../../services/auth.service';
import { AngularFireStorage } from 'angularfire2/storage';

@Component({
  selector: 'app-gallery',
  templateUrl: './gallery.component.html',
  styleUrls: ['./gallery.component.css']
})
export class GalleryComponent implements OnInit {

  //images:any = [];
  imagesURL:any = [];

  constructor(public authService:AuthService, public afAuth: AngularFireAuth, public storage: AngularFireStorage,) { }

  ngOnInit() { 
    this.authService.getUsersImages().subscribe(img => {
      let images = Object.values(img);
      images.forEach(image => {
        if(image !== 'images') { 
          this.storage.ref('/images/' + this.afAuth.auth.currentUser.uid + '/' + image).getDownloadURL().forEach(url => {
            this.imagesURL.push(url);
          });
        }  
      });
    });
  }

  // this.images.forEach(image => {
  //   console.log(image);
  //   if(image !== 'images') {
  //     this.storage.ref('/images/' + this.afAuth.auth.currentUser.uid + '/' + image).getDownloadURL().forEach(url => {
  //       console.log(url);
  //    });
  //   }
  // });

  // console.log(this.imagesURL);

}
