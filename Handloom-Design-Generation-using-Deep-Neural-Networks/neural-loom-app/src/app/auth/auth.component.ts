import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../services/auth.service';
import { FlashMessagesService } from 'angular2-flash-messages';

@Component({
  selector: 'app-auth',
  templateUrl: './auth.component.html',
  styleUrls: ['./auth.component.css']
})
export class AuthComponent implements OnInit {

  loginSelected:boolean = true;
  signupSelected:boolean = false;

  l_email:string = "";
  l_password:string = "";
  
  s_name:string = "";
  s_email:string = "";
	s_password:string = "";

  constructor(public router:Router,	public authService:AuthService,	public flashMessagesService:FlashMessagesService) { }

  ngOnInit() { }

  onLoginSelect(event: Event) {
    this.loginSelected = true;
    this.signupSelected = false;
  }

  onSignupSelect(event: Event) {
    this.signupSelected = true;
    this.loginSelected = false;
  }

  onLogin(){
    this.authService.login(this.l_email , this.l_password)
    .then((res) => {
       this.flashMessagesService.show('You are logged in now' , {cssClass: 'custom-success-alert' , timeout:4000});
        this.router.navigate(['/dashboard']);
     })
    .catch((err) => {
       this.flashMessagesService.show(err.message,{cssClass: 'custom-danger-alert' , timeout:4000});
       this.router.navigate(['/auth']);
    })
  }	

  onSignup() {
    this.authService.registerUser({"name": this.s_name, "images": []}, this.s_email , this.s_password)
      .then((res) => {
        this.flashMessagesService.show('You have registered successfully!' , {cssClass: 'custom-success-alert' , timeout:4000});
        this.router.navigate(['/dashboard']);
       })
      .catch((err) => {
        console.log(err);
        this.flashMessagesService.show(err.message,{cssClass: 'custom-danger-alert' , timeout:4000});
        this.router.navigate(['/auth']);
      })
  }	

}
