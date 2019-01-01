import { Injectable } from '@angular/core';
import { User } from '../models/user.model';
import { AngularFireAuth } from 'angularfire2/auth';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { AngularFireDatabase  , AngularFireList , AngularFireObject } from 'angularfire2/database';
import { Router } from '@angular/router';
import { FlashMessagesService } from 'angular2-flash-messages';
import { resolve } from 'url';

@Injectable({
  providedIn: 'root'
})
export class AuthService {

  currentUser:AngularFireObject<User>;
  currentUsers:AngularFireList<User>;

  constructor(public afAuth: AngularFireAuth, public router:Router, public flashMessageService: FlashMessagesService, public af: AngularFireDatabase) {   }

  login(email: string , password: string){
    return new Promise((resolve , reject) => {
      this.afAuth.auth.signInWithEmailAndPassword(email, password)
      .then(result => {
        resolve(result)
      }, err => reject(err));
    });
  }

  logout(showDefaultMessage:boolean = true){
    this.afAuth.auth.signOut();
    this.router.navigate(['/']);
    if(showDefaultMessage)
      this.flashMessageService.show("You have been logged-out successfully" , {cssClass: "custom-success-alert" , timeout:4000});
  }

  registerUser(data:User , email:string , password:string){
    this.currentUsers = this.af.list('/users') as AngularFireList<User>;

    return new Promise((resolve , reject) => {
      this.afAuth.auth.createUserAndRetrieveDataWithEmailAndPassword(email , password)
      .then(userData =>{
        resolve(userData);
        this.currentUsers.set(userData.user.uid, data)       
      }  , err => reject(err) );
    })

  }

  getUsersImages(){
    this.currentUser = this.af.object('/users/'+ this.afAuth.auth.currentUser.uid + '/images') as AngularFireObject<User>;
    return this.currentUser.snapshotChanges().pipe(map(c => ({
      id: c.payload.key , ...c.payload.val()
    })));
  }
 
  listUsers(){
    this.currentUsers = this.af.list('/users') as AngularFireList<User>;
    return this.currentUser.snapshotChanges().pipe(map(c => ({
      id: c.payload.key , ...c.payload.val()
    }))); 
  }

}
