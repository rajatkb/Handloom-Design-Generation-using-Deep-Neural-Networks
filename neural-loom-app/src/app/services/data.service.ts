import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DataService {

  constructor() { }

  private messageSource = new BehaviorSubject<Object>({"file": "", "imageSrc": "", "isMaskActive": ""});
  currentMessage = this.messageSource.asObservable();

  changeMessage(sharedData: Object) {
    this.messageSource.next({"file": sharedData["file"], "imageSrc": sharedData["imageSrc"], "isMaskActive": sharedData["isMaskActive"]});
  }
  
}
