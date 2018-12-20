import { Injectable } from '@angular/core';
import { HttpClient, HttpRequest, HttpEventType, HttpResponse } from '@angular/common/http';
import { Subject } from 'rxjs';
import { Observable } from 'rxjs';
import { User } from '../models/user.model';
import { AngularFireAuth } from 'angularfire2/auth';
import { AngularFireDatabase } from 'angularfire2/database';
import { AuthService } from './auth.service';

const BASE_URL = 'http://127.0.0.1:5000/';

@Injectable({
  providedIn: 'root'
})
export class UploadService {

  constructor(public http: HttpClient, public afAuth: AngularFireAuth) { }

  public upload(file: File)  {
    const formData: FormData = new FormData();
    formData.append('image', file);
    return this.http.post(BASE_URL + 'maskImage', formData);
  }

}
