import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { AngularFireAuth } from 'angularfire2/auth';

//const BASE_URL = 'http://35.232.10.14:7000/';
const BASE_URL = 'http://127.0.0.1:5000/';

@Injectable({
  providedIn: 'root'
})
export class UploadService {

  constructor(public http: HttpClient, public afAuth: AngularFireAuth) { }

  public uploadForGAN(file: File)  {
    const formData: FormData = new FormData();
    formData.append('image', file);
    return this.http.post(BASE_URL + 'test', formData);
  }

  public uploadForMask(file: File)  {
    const formData: FormData = new FormData();
    formData.append('image', file);
    return this.http.post(BASE_URL + 'test', formData);
  }

  public uploadForMerge(file: File, background: string, foreground: string) {
    const formData: FormData = new FormData();
    formData.append('image', file);
    formData.append('background', background);
    formData.append('foreground', foreground);
    return this.http.post(BASE_URL + 'test', formData);
  }

}
