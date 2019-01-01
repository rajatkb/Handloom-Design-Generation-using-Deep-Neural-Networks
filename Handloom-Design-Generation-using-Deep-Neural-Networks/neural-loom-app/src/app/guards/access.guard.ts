import { Injectable } from '@angular/core';
import { CanActivateChild , Router } from '@angular/router';
import { map } from 'rxjs/operators';
import { Observable } from 'rxjs';
import { FlashMessagesService } from 'angular2-flash-messages';
import { AuthService } from '../services/auth.service';
import { User } from '../models/user.model';

@Injectable()
export class AccessGuard /* implements CanActivateChild */ {

    constructor(
        private router:Router,
        public flashMessagesService:FlashMessagesService,
        public authService:AuthService
    ){}
    /*
    user:User;

    canActivateChild():Observable<boolean>{
        return this.authService.getCurrentUserDetail().map(userdata => {
            if(userdata.role_id == 1)
                return true;
            else{
                this.flashMessagesService.show("You are not the manager !!" , {cssClass: 'custom-danger-alert' , timeOut:10000})
                this.router.navigate(['/']);
                return false;
            }
        });
    }   */
} 