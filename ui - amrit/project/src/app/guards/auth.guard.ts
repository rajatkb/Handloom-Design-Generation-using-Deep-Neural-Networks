import { Injectable } from '@angular/core';
import { CanActivate , Router } from '@angular/router';
import { map } from 'rxjs/operators';
import { AngularFireAuth } from 'angularfire2/auth';
import { Observable } from 'rxjs';
import { FlashMessagesService } from 'angular2-flash-messages';

@Injectable()
export class AuthGuard implements CanActivate  {
    constructor(private router:Router, public afAuth:AngularFireAuth, public flashMessagesService:FlashMessagesService){ }
    
    canActivate(): Observable<boolean>{
       
        return this.afAuth.authState.pipe(map(auth => {
            if(!auth)
            {
                this.router.navigate(['/login']);
                this.flashMessagesService.show("Log-in to access this service", {cssClass: 'custom-danger-alert' , timeout:4000});
                return false;
            } else {
                return true;
            }
        })); 
    }

}