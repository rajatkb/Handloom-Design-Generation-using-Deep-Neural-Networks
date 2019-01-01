import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { RouterModule , Routes } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { FlashMessagesModule } from 'angular2-flash-messages';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HttpClientModule } from '@angular/common/http';

import { AngularFireModule}  from 'angularfire2';
import { AngularFireDatabaseModule , AngularFireDatabase } from 'angularfire2/database';
import { AngularFireAuthModule , AngularFireAuth } from 'angularfire2/auth';
import { AngularFirestoreModule } from 'angularfire2/firestore';
import { AngularFireStorageModule } from 'angularfire2/storage';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { AuthComponent } from './auth/auth.component';
import { DashboardComponent } from './dashboard/dashboard.component';

import { AuthService } from '../app/services/auth.service';
import { UploadService } from '../app/services/upload.service';
import { AuthGuard } from '../app/guards/auth.guard';
import { MainComponent } from './dashboard/main/main.component';
import { LandingComponent } from './dashboard/landing/landing.component';
import { GalleryComponent } from './dashboard/gallery/gallery.component';
import { FiledropDirective } from './directives/filedrop.directive';
import { DataService } from './services/data.service';

export const firebaseConfig = {
  apiKey: "AIzaSyAFkZNqoZwBOQGWIyDZ_uK_v3-HiWWzInw",
  authDomain: "project7sem-ef250.firebaseapp.com",
  databaseURL: "https://project7sem-ef250.firebaseio.com",
  projectId: "project7sem-ef250",
  storageBucket: "project7sem-ef250.appspot.com",
  messagingSenderId: "458590414263"
};

const appRoutes: Routes = [
  {path:'',component: AuthComponent},
  {path:'login',component: AuthComponent},
  {path:'signup',component:AuthComponent},
  {path:'dashboard',component:DashboardComponent, canActivate: [AuthGuard], pathMatch:'full'}  
];


@NgModule({
  declarations: [
    AppComponent,
    DashboardComponent,
    AuthComponent,
    MainComponent,
    LandingComponent,
    GalleryComponent,
    FiledropDirective
  ],
  imports: [
    BrowserModule,
    FormsModule,
    RouterModule.forRoot(appRoutes),
    AppRoutingModule,
    AngularFireModule.initializeApp(firebaseConfig),
    AngularFireDatabaseModule,
    AngularFireAuthModule,
    AngularFirestoreModule,
    AngularFireStorageModule,
    FlashMessagesModule.forRoot(),
    BrowserAnimationsModule,
    HttpClientModule
  ],
  providers: [
    AngularFireAuth,
    AngularFireDatabase,
    AuthService,
    AuthGuard,
    UploadService,
    DataService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
